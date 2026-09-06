import argparse
import json
import pickle as pkl
from copy import deepcopy
from logging import getLogger
from pathlib import Path

import numpy as np
import smplx
import tqdm
import trimesh
from scipy.spatial.transform import Rotation

from echo.configs import PreprocessConfig
from echo.data_parsers.common import (
    canonicalize_object,
    contacts_worker_sample,
    convert_segments_to_mask,
    export_sequence_to_hdf5,
    extract_motion_features,
    extract_segments,
    export_canonical_objects,
    load_canonical_transforms,
    transform_mesh,
    get_object_features,
    get_sequences_list,
    init_preprocessing,
    smooth_rigid_motion,
)
from echo.utils.parallel_map import parallel_map
from echo.utils.rotation import aa_to_matrix
from echo.utils.smpl import (
    compute_height_and_contacts,
    convert_to_neutral_beta,
    get_alignment_transform_and_align_smpl,
    load_neutral_beta_conversion,
)

logger = getLogger(__name__)


def prepare_canonical_objects_behave(cfg: PreprocessConfig):
    # convert to Path
    root_folder = Path(cfg.behave.root)
    original_objects_path = Path(cfg.behave.original_objects_path)
    target_folder = Path(cfg.env.datasets_folder) / "behave"

    # create target folder
    target_folder.mkdir(exist_ok=True, parents=True)

    # load original objects
    original_objects = {}
    for obj_path in original_objects_path.glob("*"):
        obj_name = obj_path.stem
        if obj_name in ["chairblack", "chairwood"]:
            suffix = "_f2500"
        elif obj_name == "tablesquare":
            suffix = "_f2000"
        elif obj_name == "monitor":
            suffix = "_closed_f1000"
        else:
            suffix = "_f1000"
        obj_path = obj_path / f"{obj_name}{suffix}.ply"

        obj_mesh = trimesh.load(obj_path)
        centroid = np.mean(obj_mesh.vertices, axis=0)
        # all transforms are defined w.r.t. centered object
        obj_mesh.vertices -= centroid
        original_objects[obj_name] = obj_mesh

    # canonicalize objects (reusing stored transforms when available)
    stored_transform = load_canonical_transforms(cfg, "behave")
    object_name2mesh, canonical_transform = dict(), dict()
    for obj_name, obj_mesh in tqdm.tqdm(
        original_objects.items(), total=len(original_objects), ncols=80
    ):
        if stored_transform is None:
            obj_mesh_can, R_can, t_can = canonicalize_object(obj_mesh)
        else:
            R_can, t_can = stored_transform[obj_name]
            obj_mesh_can = transform_mesh(obj_mesh, R_can, t_can)
        obj_mesh_can.visual = original_objects[obj_name].visual
        object_name2mesh[obj_name] = obj_mesh_can
        canonical_transform[obj_name] = (R_can, t_can)

    export_canonical_objects(cfg, "behave", object_name2mesh, canonical_transform)


def intersect_timestamps(timestamps_sbj, timestamps_obj, tol=1e-3):
    indices_sbj = []
    indices_obj = []
    i, j = 0, 0

    while i < len(timestamps_sbj) and j < len(timestamps_obj):
        diff = timestamps_sbj[i] - timestamps_obj[j]

        if abs(diff) <= tol:
            indices_sbj.append(i)
            indices_obj.append(j)
            i += 1
            j += 1
        elif diff < -tol:
            i += 1
        else:
            j += 1

    return np.array(indices_sbj), np.array(indices_obj)


def preprocess(cfg: PreprocessConfig):
    # convert to Path
    root_folder = Path(cfg.behave.root)
    root_converted_folder = Path(cfg.behave.root_converted)
    full_anno_path = Path(cfg.behave.full_anno_path)
    target_root = Path(cfg.env.datasets_folder) / "behave"
    hdf5_folder = target_root / f"dataset_{cfg.behave.split}_30fps"
    can_objects_path = target_root / "object_meshes"

    # create target folder
    hdf5_folder.mkdir(exist_ok=True, parents=True)

    # list dataset sequences
    all_sequences = get_sequences_list(
        "behave", full_anno_path, objects=None, subjects=None
    )

    # filter sequences based on split
    with open(cfg.behave.split_file, "r") as fp:
        split = json.load(fp)
    split_sequences = split[cfg.behave.split]
    sequences = [seq for seq in all_sequences if seq.name in split_sequences]

    # setup body models
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
            num_betas=10,
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
                cfg, gender=gender, num_betas=10
            )

    # load downsample indices
    smpl_downsample_indices = np.load(
        Path(cfg.env.assets_folder) / "smplx_downsample_indices.npy"
    )

    # preprocess each sequence
    contact_masks = {}
    for sequence in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        # load sequence info
        with (sequence / "info.json").open("r") as fp:
            sequence_info = json.load(fp)  # 'cat', 'gender'
        metadata = {
            "gender": sequence_info["gender"],
            "framerate": cfg.behave.target_fps,
        }

        # parse sequence name: Date<:02d>_Sub<:02d>_<object>_<optional:action>
        sequence_name = sequence.name
        sequence_name_split = sequence_name.split("_")
        seq_date = sequence_name_split[0]
        seq_subject = sequence_name_split[1]
        seq_action = (
            f"{sequence_name_split[3]}_{seq_date}"
            if len(sequence_name_split) == 4
            else f"{seq_date}"
        )
        seq_object = sequence_info["cat"]
        metadata["sequence_path"] = str(sequence)
        metadata["sequence_name"] = sequence_name
        metadata["object_name"] = seq_object
        metadata["subject"] = seq_subject

        # load smplx params, old smplh params and object params
        smpl_sequence = dict(np.load(root_converted_folder / f"{sequence_name}.npz"))
        obj_sequence = dict(np.load(sequence / "object_fit_all.npz"))
        original_smpl_sequence = dict(np.load(sequence / "smpl_fit_all.npz"))

        # align timestamps
        smpl_t_stamps = original_smpl_sequence["frame_times"]
        smpl_t_stamps = [float(t[1:]) for t in smpl_t_stamps]  # t0003.000
        obj_t_stamps = obj_sequence["frame_times"]
        obj_t_stamps = [float(t[1:]) for t in obj_t_stamps]  # t0003.000
        smpl_indices, obj_indices = intersect_timestamps(
            smpl_t_stamps, obj_t_stamps, tol=1e-2
        )

        # index using aligned timestamps
        T = len(smpl_t_stamps)
        smpl_sequence = {
            k: v[smpl_indices]
            if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == T
            else v
            for k, v in smpl_sequence.items()
        }
        obj_sequence = {
            k: v[obj_indices]
            if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == T
            else v
            for k, v in obj_sequence.items()
        }

        # smooth object pose, matching the body smoothing done at SMPL-H->X conversion
        if cfg.behave.object_smoothing_window > 0:
            obj_sequence["angles"], obj_sequence["trans"] = smooth_rigid_motion(
                obj_sequence["angles"],
                obj_sequence["trans"],
                cfg.behave.object_smoothing_window,
            )

        # Gather SMPL params and convert to neutral betas if needed
        T = smpl_sequence["transl"].shape[0]
        metadata["T"] = T

        smpl_params = {
            "betas": smpl_sequence["betas"].reshape(1, 10).repeat(T, 0),
            "transl": smpl_sequence["transl"],
            "global_orient": smpl_sequence["global_orient"],
            "body_pose": smpl_sequence["body_pose"],
            "left_hand_pose": smpl_sequence["left_hand_pose"],
            "right_hand_pose": smpl_sequence["right_hand_pose"],
        }
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
        R_align = Rotation.from_euler("x", -90, degrees=True).as_matrix()
        transl_align = get_alignment_transform_and_align_smpl(
            R_align, smpl_params, smpl_model
        )

        # Regenerate smpls vertices and extract hmd features
        motion_features = extract_motion_features(
            smpl_model, smpl_params, fps=cfg.behave.target_fps
        )

        # Compute object pose
        R_can, t_can = deepcopy(canonical_object_transform[seq_object])
        can_T = {"R": R_can, "t": t_can}
        obj_T = {"R": aa_to_matrix(obj_sequence["angles"]), "t": obj_sequence["trans"]}
        align_T = {"R": R_align, "t": transl_align}
        object_features = get_object_features(obj_T, can_T, align_T, "behave")

        # Pose object vertices
        obj_mesh_can = deepcopy(canonical_objects[seq_object])
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
            motion_features["smpl_joints_world"], fps=cfg.behave.target_fps
        )
        if discard_seq:
            logger.info(f"Skipping {sequence_name}: terrain interaction")
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
            "contact_threshold": cfg.behave.contact_threshold,
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
            n_jobs=36,
            tqdm_kwargs={"leave": False},
        )
        contacts_mask, sbj2obj_d = zip(*contacts_output)
        contacts_mask = np.array(contacts_mask)
        motion_features["hoi_contacts"] = np.array(sbj2obj_d)

        # 1. filter contactless interactions at the start and end of the sequence
        # 2. slice on continuous segments based on the contact mask
        segments = extract_segments(contacts_mask)
        segment_masks = convert_segments_to_mask(contacts_mask, segments)

        for segment_id, segment_mask in enumerate(segment_masks):
            contact_indices = np.where(segment_mask)[0]
            start, end = contact_indices[0], contact_indices[-1]

            seg_smpl_params = {
                key: value[start : end + 1] for key, value in smpl_params.items()
            }
            seg_motion_features = {
                key: value[start : end + 1] if key != "floor_height" else value
                for key, value in motion_features.items()
            }
            seg_obj_features = {
                key: value[start : end + 1] for key, value in object_features.items()
            }

            seg_timestamps_smpl = smpl_t_stamps[start : end + 1]
            seg_timestamps_obj = obj_t_stamps[start : end + 1]
            num_frames = end - start + 1

            # Save timestamps
            seg_motion_features["timestamps"] = seg_timestamps_smpl
            seg_obj_features["timestamps_obj"] = seg_timestamps_obj

            # Update metadata
            metadata["action"] = (f"{seq_action}_seg{segment_id:02d}").encode("utf-8")
            metadata["t_start"] = seg_timestamps_smpl[0]
            metadata["t_end"] = seg_timestamps_smpl[-1]
            metadata["T"] = num_frames

            # save to hdf5
            h5_name = (
                hdf5_folder
                / f"{seq_subject}_{seq_object}_{seq_action}_seg{segment_id:02d}.hdf5"
            )
            export_sequence_to_hdf5(
                seg_smpl_params,
                seg_motion_features,
                seg_obj_features,
                metadata,
                h5_name,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess BEHAVE data for MoSynt from sparse input"
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

    # preprocess data
    if arguments.generate_objects:
        prepare_canonical_objects_behave(config)

    preprocess(config)
