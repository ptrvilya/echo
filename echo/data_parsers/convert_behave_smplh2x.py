"""Convert BEHAVE sequences from SMPL-H to SMPL-X format using smplfitter.
"""
import argparse
import json
import logging
import os
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import tqdm

from echo.configs import PreprocessConfig
from echo.data_parsers.common import init_preprocessing, smooth_rigid_motion

logger = getLogger(__name__)

# SMPL-X pose_rotvecs layout (55 joints x 3):
# global(1) body(21) jaw(1) leye(1) reye(1) left_hand(15) right_hand(15)
GLOBAL_ORIENT = slice(0, 3)
BODY_POSE = slice(3, 66)
FACE_POSE = slice(66, 75)
LEFT_HAND_POSE = slice(75, 120)
RIGHT_HAND_POSE = slice(120, 165)


def smooth_body_pose(pose_rotvecs: np.ndarray, transl: np.ndarray, window: int):
    """Smooth global orient + body rotations and translation; hands/face untouched."""
    smoothed, transl = smooth_rigid_motion(
        pose_rotvecs[:, : BODY_POSE.stop], transl, window
    )
    pose_rotvecs = pose_rotvecs.copy()
    pose_rotvecs[:, : BODY_POSE.stop] = smoothed
    return pose_rotvecs, transl


def build_models(gender: str, device: str):
    from smplfitter.pt import BodyConverter, BodyFitter, BodyModel

    bm_smplh = BodyModel("smplh", gender, num_betas=10).to(device)
    bm_smplx = BodyModel("smplx", gender, num_betas=10).to(device)
    converter = BodyConverter(bm_smplh, bm_smplx).to(device)
    fitter = BodyFitter(bm_smplx).to(device)
    return bm_smplh, bm_smplx, converter, fitter


@torch.inference_mode()
def convert_sequence(
    cfg: PreprocessConfig, models, smplh_sequence: dict, device: str
):
    bm_smplh, bm_smplx, converter, fitter = models
    poses = torch.from_numpy(smplh_sequence["poses"]).float()
    betas = torch.from_numpy(smplh_sequence["betas"]).float()
    trans = torch.from_numpy(smplh_sequence["trans"]).float()
    T = poses.shape[0]
    batch_size = cfg.behave.smplfitter_batch_size

    # pose SMPL-H and remap vertices to SMPL-X topology
    target_vertices = []
    for start in range(0, T, batch_size):
        sl = slice(start, start + batch_size)
        smplh_out = bm_smplh(
            pose_rotvecs=poses[sl].to(device),
            shape_betas=betas[sl].to(device),
            trans=trans[sl].to(device),
        )
        target_vertices.append(converter.convert_vertices(smplh_out["vertices"]).cpu())
    target_vertices = torch.cat(target_vertices, dim=0)

    # fit a single set of betas shared across a subsample of frames
    shape_indices = np.unique(
        np.linspace(0, T - 1, min(T, cfg.behave.smplfitter_max_shape_frames)).round()
    ).astype(int)
    shape_fit = fitter.fit(
        target_vertices=target_vertices[shape_indices].to(device),
        share_beta=True,
        num_iter=cfg.behave.smplfitter_num_iter,
        beta_regularizer=0.0,
        final_adjust_rots=False,
        requested_keys=["shape_betas"],
    )
    shared_betas = shape_fit["shape_betas"][:1]  # (1, 10)

    # fit per-frame pose and translation with the shared betas fixed
    pose_rotvecs, transl = [], []
    for start in range(0, T, batch_size):
        batch_target = target_vertices[start : start + batch_size].to(device)
        pose_fit = fitter.fit_with_known_shape(
            shape_betas=shared_betas.expand(batch_target.shape[0], -1),
            target_vertices=batch_target,
            num_iter=cfg.behave.smplfitter_num_iter,
            requested_keys=["pose_rotvecs"],
        )
        # SMPL-H carries no face signal, zero out jaw/eye rotations. BEHAVE hand
        # poses are a constant relaxed pose per sequence with no real articulation
        # signal, so keep them verbatim instead of the (noisy) per-frame hand fit.
        batch_pose = pose_fit["pose_rotvecs"]
        batch_pose[:, FACE_POSE] = 0.0
        batch_pose[:, LEFT_HAND_POSE] = poses[start : start + batch_size, 66:111].to(device)
        batch_pose[:, RIGHT_HAND_POSE] = poses[start : start + batch_size, 111:156].to(device)

        pose_rotvecs.append(batch_pose.cpu())
        transl.append(pose_fit["trans"].cpu())
    pose_rotvecs = torch.cat(pose_rotvecs, dim=0).numpy().astype(np.float32)
    transl = torch.cat(transl, dim=0).numpy().astype(np.float32)

    if cfg.behave.smplfitter_smoothing_window > 0:
        pose_rotvecs, transl = smooth_body_pose(
            pose_rotvecs, transl, cfg.behave.smplfitter_smoothing_window
        )

    # validate the final (possibly smoothed) parameters against the target mesh
    v2v_sum = 0.0
    pose_t, transl_t = torch.from_numpy(pose_rotvecs), torch.from_numpy(transl)
    for start in range(0, T, batch_size):
        sl = slice(start, start + batch_size)
        smplx_out = bm_smplx(
            pose_rotvecs=pose_t[sl].to(device),
            shape_betas=shared_betas.expand(pose_t[sl].shape[0], -1),
            trans=transl_t[sl].to(device),
        )
        distances = torch.norm(smplx_out["vertices"] - target_vertices[sl].to(device), dim=-1)
        v2v_sum += distances.mean(dim=1).sum().item()
    v2v_error = v2v_sum / T

    return pose_rotvecs, transl, shared_betas[0].cpu().numpy().astype(np.float32), v2v_error


def convert(cfg: PreprocessConfig):
    # BodyConverter and BodyModel resolve assets via $DATA_ROOT/body_models
    os.environ["DATA_ROOT"] = cfg.behave.smplfitter_data_root
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_folder = Path(cfg.behave.smplfitter_output)
    output_folder.mkdir(parents=True, exist_ok=True)

    sequences = sorted(Path(cfg.behave.full_anno_path).glob("*/smpl_fit_all.npz"))
    if len(cfg.behave.smplfitter_sequences) > 0:
        sequences = [
            s for s in sequences if s.parent.name in cfg.behave.smplfitter_sequences
        ]
    logger.info(f"Converting {len(sequences)} sequences to {output_folder}")

    models_per_gender = {}
    v2v_errors = {}
    for params_path in tqdm.tqdm(sequences, total=len(sequences), ncols=80):
        sequence_name = params_path.parent.name
        output_path = output_folder / f"{sequence_name}.npz"
        if cfg.behave.smplfitter_skip_existing and output_path.exists():
            continue

        with (params_path.parent / "info.json").open("r") as fp:
            gender = json.load(fp)["gender"]
        if gender not in models_per_gender:
            models_per_gender[gender] = build_models(gender, device)

        smplh_sequence = dict(np.load(params_path))
        pose_rotvecs, transl, betas, v2v_error = convert_sequence(
            cfg, models_per_gender[gender], smplh_sequence, device
        )
        v2v_errors[sequence_name] = v2v_error
        logger.info(f"{sequence_name}: mean v2v error {v2v_error * 1000:.2f} mm")

        T = pose_rotvecs.shape[0]
        frame_times = np.array(
            [float(t[1:]) for t in smplh_sequence["frame_times"]], dtype=np.float64
        )
        frame_times = frame_times - frame_times[0]
        if T > 1:
            frame_rate = round(1.0 / float(np.median(np.diff(frame_times))))
        else:
            frame_rate = cfg.behave.target_fps

        np.savez(
            output_path,
            global_orient=pose_rotvecs[:, GLOBAL_ORIENT],
            body_pose=pose_rotvecs[:, BODY_POSE],
            left_hand_pose=pose_rotvecs[:, LEFT_HAND_POSE],
            right_hand_pose=pose_rotvecs[:, RIGHT_HAND_POSE],
            transl=transl,
            betas=betas,
            jaw_pose=np.zeros((T, 3), dtype=np.float64),
            leye_pose=np.zeros((T, 3), dtype=np.float64),
            reye_pose=np.zeros((T, 3), dtype=np.float64),
            expression=np.zeros((T, 10), dtype=np.float64),
            gender=gender,
            frame_times=frame_times,
            save_name=sequence_name,
            motion_info=np.array(["", "", sequence_name, "", "", ""]),
            mocap_frame_rate=frame_rate,
        )

    if len(v2v_errors) > 0:
        errors = np.array(list(v2v_errors.values())) * 1000
        worst = max(v2v_errors, key=lambda k: v2v_errors[k])
        logger.info(
            f"Converted {len(v2v_errors)} sequences: "
            f"mean v2v {errors.mean():.2f} mm, "
            f"max v2v {errors.max():.2f} mm ({worst})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BEHAVE sequences from SMPL-H to SMPL-X with smplfitter"
    )

    parser.add_argument(
        "--config", "-c", type=str, nargs="*", help="Path to YAML config(-s) file."
    )
    parser.add_argument(
        "overrides", type=str, nargs="*", help="Overrides for the config."
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = init_preprocessing(arguments)

    convert(config)
