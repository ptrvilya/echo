import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import smplx
import tqdm

from echo.configs import PreprocessConfig
from echo.utils.smpl import (
    compute_height_and_contacts,
    convert_to_neutral_beta,
    load_neutral_beta_conversion,
    vposer2smplx_keys,
)

from .common import export_sequence_to_hdf5, extract_motion_features, init_preprocessing

logger = logging.getLogger(__name__)


def check_skip(path_name: str) -> bool:
    """Copied conditions from https://github.com/davrempe/humor/blob/main/humor/scripts/cleanup_amass_data.py"""
    if "BioMotionLab_NTroje" in path_name and (
        "treadmill" in path_name or "normal_" in path_name
    ):
        return True
    if "MPI_HDM05" in path_name and "dg/HDM_dg_07-01" in path_name:
        return True
    return False


def load_smpl_params(
    filepath: str, target_framerate: int
) -> Tuple[Dict[str, npt.NDArray], Dict[str, Any]]:
    # load data
    # keys are ['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 'markers_latent', 'latent_labels',
    # 'markers_latent_vids', 'trans', 'poses', 'betas', 'num_betas', 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']
    data = np.load(filepath, allow_pickle=True)

    # check if framerate is present
    framerate = data.get("mocap_frame_rate", None)

    # correct mislabeled data
    if filepath.find("BMLhandball") >= 0:
        framerate = 240
    elif filepath.find("20160930_50032") >= 0 or filepath.find("20161014_50033") >= 0:
        framerate = 59

    if framerate is None:
        return dict(), dict()

    # Read data
    smpl_params = dict()
    gender = data["gender"].item()
    for key in [
        "root_orient",
        "pose_body",
        "pose_hand",
        "pose_jaw",
        "pose_eye",
        "trans",
        "betas",
    ]:
        if key not in data:
            logger.warning(f"Key {key} not found in {filepath}")
            continue
        smpl_params[key] = data[key]

    # check correctness
    if "pose_body" not in smpl_params:
        return dict(), dict()

    # Downsample to target framerate
    T = smpl_params["pose_body"].shape[0]
    framerate_ratio = float(target_framerate) / framerate
    downsample_indices = np.linspace(
        0, T - 1, num=int(T * framerate_ratio), dtype=np.int32
    )
    for key in smpl_params.keys():
        if key == "betas":
            continue
        smpl_params[key] = smpl_params[key][downsample_indices, ...]

    # Only keep middle 80% of sequences to avoid redundant static poses
    T = smpl_params["pose_body"].shape[0]
    s_idx, e_idx = int(0.1 * T), int(0.9 * T)
    T = e_idx - s_idx

    # Discard too short sequences
    if T <= target_framerate:
        return dict(), dict()

    # Trim sequences
    for key in smpl_params.keys():
        if key == "betas":
            continue
        smpl_params[key] = smpl_params[key][s_idx:e_idx, ...]

    # Create metadata
    metadata = {"gender": gender, "T": T, "framerate": target_framerate}

    return smpl_params, metadata


def preprocess(cfg: PreprocessConfig):
    # select subsets based on split
    assert cfg.amass.split in [
        "train",
        "val",
        "test",
    ], f"Unknown split {cfg.amass.split}"
    if cfg.amass.split == "train":
        subsets = cfg.amass.subsets_train
    elif cfg.amass.split == "val":
        subsets = cfg.amass.subsets_val
    elif cfg.amass.split == "test":
        subsets = cfg.amass.subsets_test

    # setup body models
    gendered_bm = {}
    if cfg.common.convert_to_neutral:
        genders = ["neutral"]
    else:
        genders = ["male", "female", "neutral"]
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

    # load neutral betas conversion
    neutral_conversion = dict()
    if cfg.common.convert_to_neutral:
        for gender in ["male", "female"]:
            neutral_conversion[gender] = load_neutral_beta_conversion(
                cfg, gender=gender, num_betas=16
            )

    # init folder structure
    amass_root = Path(cfg.amass.root)
    root_folder = Path(cfg.env.datasets_folder) / "amass"
    target_folder_name = f"dataset_{cfg.amass.split}_{cfg.amass.target_fps}fps"
    target_folder = root_folder / target_folder_name
    target_folder.mkdir(exist_ok=True, parents=True)

    # main preprocessing loop
    skipped = []
    for subset in tqdm.tqdm(subsets, ncols=80):
        # get filepaths from the AMASS dataset
        filepaths = (amass_root / subset).rglob("**/*.npz")
        filepaths = [str(filepath.relative_to(amass_root)) for filepath in filepaths]

        for filepath in tqdm.tqdm(filepaths, ncols=80, leave=False):
            # Skip problematic sequences
            if check_skip(str(filepath)):
                logger.info(f"Skipping {filepath}: bad sequence")
                skipped.append(filepath)
                continue

            # load smpl params
            bm_params, metadata = load_smpl_params(
                str(amass_root / filepath), cfg.amass.target_fps
            )
            if len(bm_params) == 0:
                logger.info(f"Skipping {filepath}: no bm params")
                skipped.append(filepath)
                continue

            # convert to neutral betas
            if cfg.common.convert_to_neutral and metadata["gender"] != "neutral":
                bm_params["betas"] = convert_to_neutral_beta(
                    bm_params["betas"], *neutral_conversion[metadata["gender"]]
                )
                metadata["gender"] = "neutral"

            # forward pass body model and extract motion features
            bm = gendered_bm[metadata["gender"]]
            bm_params["betas"] = np.repeat(
                bm_params["betas"][None], metadata["T"], axis=0
            )
            bm_params = vposer2smplx_keys(bm_params)
            motion_features = extract_motion_features(
                bm, bm_params, fps=cfg.amass.target_fps
            )
            _ = motion_features.pop("sbj_vertices")

            # Convert to numpy
            for key in motion_features.keys():
                motion_features[key] = motion_features[key].cpu().numpy()

            # Compute height and env_contacts
            floor_height, env_contacts, discard_seq = compute_height_and_contacts(
                motion_features["smpl_joints_world"], fps=cfg.amass.target_fps
            )
            if discard_seq:
                logger.info(f"Skipping {filepath}: terrain interaction")
                skipped.append(filepath)
                continue
            metadata["floor_height"] = floor_height
            motion_features["env_contacts"] = env_contacts

            # Discard edge frames
            metadata["T"] -= 2
            for key in bm_params.keys():
                bm_params[key] = bm_params[key][1:-1]

            # Assemble metadata
            metadata["filepath"] = filepath
            metadata["subset"] = subset

            # Construct timesatamps for compatibility
            motion_features["timestamps"] = (
                np.arange(metadata["T"]) / cfg.amass.target_fps
            )

            # save to hdf5
            seq_name = filepath.replace("/", "\\")
            seq_name = seq_name.replace(".npz", "")
            h5_name = target_folder / f"{seq_name}.hdf5"

            export_sequence_to_hdf5(
                bm_params,
                motion_features,
                dict(),
                metadata,
                h5_name,
            )

    # save info about skipped sequences
    with open(
        target_folder
        / f"skipped_files_{cfg.amass.split}_{cfg.amass.target_fps}fps.txt",
        "w",
    ) as f:
        for filepath in skipped:
            f.write(f"{filepath}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess AMASS data for MoSynt from sparse input"
    )

    parser.add_argument(
        "--config", "-c", type=str, nargs="*", help="Path to YAML config(-s) file."
    )
    parser.add_argument(
        "overrides", type=str, nargs="*", help="Overrides for the config."
    )
    arguments = parser.parse_args()

    config = init_preprocessing(arguments)

    # preprocess data
    preprocess(config)
