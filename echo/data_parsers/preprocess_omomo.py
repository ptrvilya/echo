import argparse
import pickle as pkl
from logging import getLogger
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch
import tqdm
import trimesh
from scipy.spatial.transform import Rotation

from echo.configs import PreprocessConfig
from echo.data_parsers.common import (
    contacts_worker_sample,
    export_sequence_to_hdf5,
    extract_motion_features,
    export_canonical_objects,
    load_canonical_transforms,
    get_object_features,
    get_sequences_list,
    init_preprocessing,
)
from echo.utils.parallel_map import parallel_map
from echo.utils.smpl import (
    compute_height_and_contacts,
    convert_to_neutral_beta,
    get_alignment_transform_and_align_smpl,
    load_neutral_beta_conversion,
)

logger = getLogger(__name__)


def prepare_canonical_objects_omomo(cfg: PreprocessConfig):
    # convert to Path
    raw_objects_path = Path(cfg.omomo.raw_objects_path)
    target_folder = Path(cfg.env.datasets_folder) / "omomo"

    # create target folder
    target_folder.mkdir(exist_ok=True, parents=True)

    # canonical transform: stored one, otherwise the dataset's orig2can
    canonical_transform = load_canonical_transforms(cfg, "omomo")
    if canonical_transform is None:
        with open(cfg.omomo.canonical_transform_path, "rb") as fp:
            orig_canonical_transform = pkl.load(fp)
        canonical_transform = {
            cfg.omomo.object_name_mapping.get(obj_name, obj_name): (
                transform["R"],
                transform["t"],
                transform["scale"],
            )
            for obj_name, transform in orig_canonical_transform.items()
        }

    # canonicalize the raw meshes: (R @ v + t) * scale
    # (the *_bottom / *_top part meshes are covered by the combined files)
    object_name2mesh = dict()
    for obj_path in sorted(raw_objects_path.glob("*_cleaned_simplified.obj")):
        obj_name = obj_path.stem.removesuffix("_cleaned_simplified")
        mapped_obj_name = cfg.omomo.object_name_mapping.get(obj_name, obj_name)
        mesh = trimesh.load(obj_path, process=False, force="mesh")
        R, t, scale = canonical_transform[mapped_obj_name]
        vertices = (mesh.vertices @ np.asarray(R).T + np.asarray(t)) * np.float64(scale)
        object_name2mesh[mapped_obj_name] = trimesh.Trimesh(
            vertices, mesh.faces, process=False
        )
    canonical_transform = {k: canonical_transform[k] for k in object_name2mesh}

    export_canonical_objects(cfg, "omomo", object_name2mesh, canonical_transform)


def preprocess(cfg: PreprocessConfig):
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    # convert to Path
    root_folder = Path(cfg.omomo.root)
    anno_path = Path(cfg.omomo.anno_path)
    target_root = Path(cfg.env.datasets_folder) / "omomo"
    hdf5_folder = target_root / f"dataset_{cfg.omomo.split}_{cfg.omomo.target_fps}fps"
    can_objects_path = target_root / "object_meshes"

    # create target folder
    hdf5_folder.mkdir(exist_ok=True, parents=True)

    # list dataset sequences
    all_sequences = get_sequences_list(
        "omomo", anno_path, objects=None, subjects=config.omomo.subjects
    )

    # setup body models
    device = torch.device("cuda")
    gendered_bm = {}
    if cfg.common.convert_to_neutral:
        genders = ["neutral"]
    else:
        genders = ["male", "female"]
    for gender in genders:
        gendered_bm[gender] = smplx.SMPLX(
            model_path=cfg.env.smplx_folder,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_betas=16,
            batch_size=1,
            flat_hand_mean=True,
        ).cuda()

    # Load canonical objects, and canonical transform
    canonical_objects = dict()
    for obj_path in can_objects_path.glob("*.ply"):
        obj_name = obj_path.stem
        obj_mesh = trimesh.load(obj_path)
        canonical_objects[obj_name] = obj_mesh
    with (target_root / "canonical_object_transform.pkl").open("rb") as fp:
        canonical_object_transform = pkl.load(fp)

    # load neutral betas conversion
    neutral_conversion = dict()
    if cfg.common.convert_to_neutral:
        for gender in ["male", "female"]:
            neutral_conversion[gender] = load_neutral_beta_conversion(
                cfg, gender=gender, num_betas=16
            )

    # load smpl downsampling for contacts
    smpl_downsample_indices = np.load(
        Path(cfg.env.assets_folder) / "smplx_downsample_indices.npy"
    )

    # load annotations file
    annotations = joblib.load(anno_path)

    # preprocess each sequence
    contact_masks = {}
    for sequence in tqdm.tqdm(all_sequences, total=len(all_sequences), ncols=80):
        # load sequence info
        seq_name = annotations[sequence]["seq_name"]
        seq_name_split = seq_name.split("_")
        seq_subject = seq_name_split[0]
        seq_object = seq_name_split[1]
        seq_object = cfg.omomo.object_name_mapping.get(seq_object, seq_object)
        seq_action = seq_name_split[2]
        seq_gender = annotations[sequence]["gender"].item()

        # setup metadata
        metadata = {
            "sequence_path": str(sequence),
            "sequence_name": seq_name,
            "object_name": seq_object,
            "subject": seq_subject,
            "action": seq_action,
            "gender": seq_gender,
        }

        # load smplx params
        T = len(annotations[sequence]["trans"])
        metadata["T"] = T
        smpl_params = {
            "betas": annotations[sequence]["betas"].reshape(1, 16).repeat(T, 0),
            "transl": annotations[sequence]["trans"],
            "global_orient": annotations[sequence]["root_orient"].reshape(T, 3),
            "body_pose": annotations[sequence]["pose_body"].reshape(T, 21 * 3),
        }
        smpl_params.update(
            {
                "left_hand_pose": np.zeros((T, 15 * 3), dtype=np.float32),
                "right_hand_pose": np.zeros((T, 15 * 3), dtype=np.float32),
            }
        )

        # convert to neutral betas
        if cfg.common.convert_to_neutral and metadata["gender"] != "neutral":
            smpl_params["betas"] = convert_to_neutral_beta(
                smpl_params["betas"], *neutral_conversion[metadata["gender"]]
            )
            metadata["gender"] = "neutral"

        # Select smpl model and obtain alignment transform, such that
        #   rotation to align axes (z-up)
        #   align the ground plane with feet
        #   align the center of the body with (0,0) on (X,Y)
        smpl_model = gendered_bm[metadata["gender"]]
        R_align = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        transl_align = get_alignment_transform_and_align_smpl(
            R_align, smpl_params, smpl_model
        )

        # Regenerate smpls vertices and extract hmd features
        motion_features = extract_motion_features(
            smpl_model, smpl_params, fps=cfg.omomo.target_fps
        )

        # Compute object pose
        R_can, t_can, s_can = canonical_object_transform[seq_object]
        can_T = {"R": R_can, "t": t_can, "s": s_can}
        obj_T = {
            "R": annotations[sequence]["obj_rot"],
            "t": annotations[sequence]["obj_trans"].reshape(-1, 3),
        }
        align_T = {"R": R_align, "t": transl_align}
        object_features = get_object_features(obj_T, can_T, align_T, "omomo")

        # load canonical mesh
        obj_mesh_can = canonical_objects[seq_object]

        # Sample and pose object vertices
        obj_points_can = (
            np.array(obj_mesh_can.sample(4000))
            .reshape(1, -1, 3)
            .repeat(metadata["T"], 0)
        )
        obj_points_seq = np.matmul(
            object_features["obj_R"], obj_points_can.transpose(0, 2, 1)
        ).transpose(0, 2, 1) + object_features["obj_transl"].reshape(
            metadata["T"], 1, 3
        )

        # Convert to numpy
        for key in motion_features.keys():
            motion_features[key] = motion_features[key].cpu().numpy()

        # Compute height and env_contacts
        floor_height, env_contacts, discard_seq = compute_height_and_contacts(
            motion_features["smpl_joints_world"], fps=cfg.omomo.target_fps
        )
        if discard_seq:
            logger.info(f"Skipping {seq_name}: terrain interaction")
            continue
        metadata["floor_height"] = floor_height
        motion_features["env_contacts"] = env_contacts

        # Discard edge frames
        metadata["T"] -= 2
        for key in smpl_params.keys():
            smpl_params[key] = smpl_params[key][1:-1]
        for key in object_features.keys():
            object_features[key] = object_features[key][1:-1]
        obj_points_seq = obj_points_seq[1:-1]

        # Compute H-O contacts
        sbj_vertices = motion_features.pop("sbj_vertices")
        contacts_shared_args = {
            "contact_threshold": cfg.omomo.contact_threshold,
            "smpl_downsample_indices": smpl_downsample_indices,
        }
        contacts_input = [
            {
                "obj_points": obj_points_seq[t],
                "sbj_verts": sbj_vertices[t],
            }
            for t in range(metadata["T"])
        ]
        contacts_output = parallel_map(
            contacts_input,
            contacts_worker_sample,
            const_args=contacts_shared_args,
            use_kwargs=True,
            n_jobs=24,
            tqdm_kwargs={"leave": False},
        )
        contacts_mask, sbj2obj_d = zip(*contacts_output)
        contacts_mask = np.array(contacts_mask)
        motion_features["hoi_contacts"] = np.array(sbj2obj_d)

        # filter contactless interactions at the start and end of the sequence
        contact_indices = np.where(contacts_mask)[0]
        if len(contact_indices) < 50:
            logger.info(f"Skipping {seq_name}: too few contacts")
            continue

        # Prepare data for saving
        start, end = contact_indices[0], contact_indices[-1]
        smpl_params = {
            key: value[start : end + 1] for key, value in smpl_params.items()
        }
        motion_features = {
            key: value[start : end + 1] if key != "floor_height" else value
            for key, value in motion_features.items()
        }
        object_features = {
            key: value[start : end + 1] for key, value in object_features.items()
        }
        metadata["T"] = end - start + 1
        metadata["t_start"] = 0
        metadata["t_end"] = metadata["T"]
        motion_features["timestamps"] = (
            np.arange(0, metadata["T"]) / cfg.omomo.target_fps
        )

        # save to hdf5
        sequence_name = f"{seq_subject}_{seq_object}_{seq_action}"
        h5_name = hdf5_folder / f"{sequence_name}.hdf5"
        export_sequence_to_hdf5(
            smpl_params,
            motion_features,
            object_features,
            metadata,
            h5_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess OMOMO data for MoSynt from sparse input"
    )

    parser.add_argument(
        "--config", "-c", type=str, nargs="*", help="Path to YAML config(-s) file."
    )
    parser.add_argument(
        "--generate-objects",
        "-g",
        action="store_true",
        help="Generate canonical objects.",
    )
    parser.add_argument(
        "overrides", type=str, nargs="*", help="Overrides for the config."
    )
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)

    # define split
    if len(config.omomo.subjects) == 0:
        if config.omomo.split == "train":
            config.omomo.subjects = [f"sub{s}" for s in range(1, 16)]
        else:
            config.omomo.subjects = ["sub16", "sub17"]

    # preprocess data
    if arguments.generate_objects:
        prepare_canonical_objects_omomo(config)

    preprocess(config)
