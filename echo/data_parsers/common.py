import pickle as pkl
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import joblib
import numpy as np
import numpy.typing as npt
import torch
import trimesh
from omegaconf import OmegaConf
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from typing_extensions import overload

from echo.configs import PreprocessConfig
from echo.utils.rotation import (
    aa_to_matrix,
    matrix_to_aa,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from echo.utils.smpl import (
    EYEJAW_END,
    EYEJAW_START,
    SMPL_JOINT_TO_ID,
    infer_smpl,
    local2global_pose,
)
from echo.utils.transforms import SE3, SO3


def init_preprocessing(arguments) -> PreprocessConfig:
    # create default config
    structured_config = PreprocessConfig()

    # convert to omegaconf
    config = OmegaConf.structured(structured_config)

    # merge with config(-s) from file
    if arguments.config is not None:
        for config_file in arguments.config:
            config = OmegaConf.merge(config, OmegaConf.load(config_file))

    # merge with cli
    if len(arguments.overrides) > 0:
        cli_config = OmegaConf.from_dotlist(arguments.overrides)
        config = OmegaConf.merge(config, cli_config)

    OmegaConf.resolve(config)

    return config


def contacts_worker_sample(
    obj_points: npt.NDArray,
    sbj_verts: npt.NDArray,
    smpl_downsample_indices: npt.NDArray,
    contact_threshold: float,
):
    sbj2obj_d = cdist(sbj_verts, obj_points).min(axis=1)

    contacts_map = sbj2obj_d < contact_threshold
    contacts_flag = np.any(contacts_map)
    sbj2obj_d = sbj2obj_d[smpl_downsample_indices]

    return contacts_flag, sbj2obj_d


def get_sequences_list(dataset, input_path, subjects=None, objects=None):
    all_seqs = []
    subjects = ["*"] if subjects is None else subjects
    if dataset == "behave":
        # behave-30fps-params-v1 / Date<:02d>_Sub<:02d>_<object>_<optional:action>
        if objects is not None and len(objects) > 0:
            for sbj in subjects:
                for object in objects:
                    all_seqs.extend(list(input_path.glob(f"*_{sbj}_{object}*/")))
        else:
            for sbj in subjects:
                all_seqs.extend(list(input_path.glob(f"*_{sbj}_*/")))
    elif dataset == "omomo":
        annotations = joblib.load(input_path)
        seq_ids = list(annotations.keys())

        for seq_id in seq_ids:
            seq_name = annotations[seq_id]["seq_name"]
            seq_name_split = seq_name.split("_")
            seq_sbj = seq_name_split[0]
            seq_object = seq_name_split[1]

            if objects is not None and len(objects) > 0:
                if seq_sbj in subjects and seq_object in objects:
                    all_seqs.append(seq_id)
            elif seq_sbj in subjects:
                all_seqs.append(seq_id)
    return all_seqs


def smooth_rigid_motion(rotvecs: npt.NDArray, trans: npt.NDArray, window: int):
    """Savitzky-Golay smoothing of axis-angle rotations (T, n*3), done on per-joint
    quaternions with hemisphere alignment, and of translations (T, 3)."""
    T = rotvecs.shape[0]
    window = min(window if window % 2 == 1 else window - 1, T if T % 2 == 1 else T - 1)
    if window < 5:
        return rotvecs, trans

    n = rotvecs.shape[1] // 3
    quats = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(T, n, 4)
    dots = np.sum(quats[1:] * quats[:-1], axis=-1)
    signs = np.cumprod(np.where(dots < 0, -1.0, 1.0), axis=0)
    quats[1:] *= signs[..., None]
    quats = savgol_filter(quats, window, polyorder=2, axis=0)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
    smoothed = Rotation.from_quat(quats.reshape(-1, 4)).as_rotvec().reshape(T, -1)

    trans_smoothed = savgol_filter(trans, window, polyorder=2, axis=0)
    return smoothed.astype(rotvecs.dtype), trans_smoothed.astype(trans.dtype)


def estimate_velocity(data_seq: npt.NDArray, h: float) -> npt.NDArray:
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py

    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.

    Parameters:
        data_seq: (T, ...)
        h: step size (1 / fps)
    Returns:
        v: (T-2, ...)
    """
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2 * h)
    return data_vel_seq


def estimate_angular_velocity(rot_seq: npt.NDArray, h: float) -> npt.NDArray:
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py

    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)

    Parameters:
        rot_seq: (T, ..., 3, 3)
        h: step size (1 / fps)
    Returns:
        w: (T-2, ..., 3)

    """
    # see https://en.wikipedia.org/wiki/Angular_velocity_tensor
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)

    return w


def smpl_joints_to_motion_features(
    joints_th: torch.Tensor,
    bm_parents_th: torch.Tensor,
    bm_params_th: Dict[str, torch.Tensor],
    fps: int = 30,
):
    HEAD_ID, LHAND_ID, RHAND_ID = (
        SMPL_JOINT_TO_ID["head"],
        SMPL_JOINT_TO_ID["left_wrist"],
        SMPL_JOINT_TO_ID["right_wrist"],
    )
    SPARSE_IDS = [HEAD_ID, LHAND_ID, RHAND_ID]

    # Get sequence length
    T = bm_params_th["body_pose"].shape[0]

    # convert pose from aa to 6d
    pose_aa_th = torch.cat(
        [
            bm_params_th["global_orient"],
            bm_params_th["body_pose"],
            torch.zeros((T, 3), dtype=torch.float),
            torch.zeros((T, 3), dtype=torch.float),
            torch.zeros((T, 3), dtype=torch.float),
            bm_params_th["left_hand_pose"],
            bm_params_th["right_hand_pose"],
        ],
        dim=1,
    )
    pose_aa_local = pose_aa_th.reshape(-1, 3)
    pose_matrix_local = aa_to_matrix(pose_aa_local)
    pose_6d_local = matrix_to_rotation_6d(pose_matrix_local).reshape(T, -1, 6)

    # conver to global rotmats (relative to the origin)
    pose_matrix_global = local2global_pose(
        pose_matrix_local.reshape(T, -1, 3, 3), bm_parents_th
    ).reshape(T, -1, 3, 3)

    # Remove eye and jaw poses
    pose_6d_local = torch.cat(
        [pose_6d_local[:, :EYEJAW_START], pose_6d_local[:, EYEJAW_END:]], dim=1
    )
    pose_matrix_global = torch.cat(
        [pose_matrix_global[:, :EYEJAW_START], pose_matrix_global[:, EYEJAW_END:]],
        dim=1,
    )

    # Convert head and hands poses to SE3
    T_world_head = (
        SE3.from_rotation_and_translation(
            SO3.from_matrix(pose_matrix_global[:, [HEAD_ID], :, :]),
            joints_th[:, [HEAD_ID]],
        )
        .parameters()
        .clone()
    )
    T_world_hands = (
        SE3.from_rotation_and_translation(
            SO3.from_matrix(pose_matrix_global[:, [LHAND_ID, RHAND_ID], :, :]),
            joints_th[:, [LHAND_ID, RHAND_ID]],
        )
        .parameters()
        .clone()
    )

    # Compute relative transformation between previous frame and current frame
    # for head and hands
    T_head_tm1_head_t = (
        SE3(T_world_head[:-1].clone()).inverse() @ SE3(T_world_head[1:].clone())
    ).parameters()[1:]
    T_hands_tm1_hands_t = (
        SE3(T_world_hands[:-1].clone()).inverse() @ SE3(T_world_hands[1:].clone())
    ).parameters()[1:]

    # Compute rotation-based features
    # Global rotation and angular velocity (in 3d vector form)
    pose_6d_global = matrix_to_rotation_6d(pose_matrix_global).reshape(T, -1, 6)
    sparse_R_6d_global = pose_6d_global[1:-1, SPARSE_IDS, :]
    sparse_w_global = torch.from_numpy(
        estimate_angular_velocity(pose_matrix_global[:, SPARSE_IDS].numpy(), 1 / fps),
    ).float()

    # Compute joint-based features
    # Global coordinate and linear velocity
    joints_position_global = joints_th[:, :EYEJAW_START, :]
    sparse_XYZ_global = joints_position_global[:, SPARSE_IDS, :]
    sparse_V_global = torch.from_numpy(
        estimate_velocity(sparse_XYZ_global.numpy(), 1 / fps)
    ).float()

    # Compute body and hand poses in 6d for supervision
    # cropped [1:-1] in time to stay aligned with the velocity-based features
    body_pose_6d = pose_6d_local[1:-1, 1:22]
    hand_pose_6d = pose_6d_local[1:-1, 22:52]

    return {
        # transformation from world to head
        "T_world_head": T_world_head[1:-1],
        # transformation from world to hands
        "T_world_hands": T_world_hands[1:-1],
        # transformation from head on previous frame to head on current frame
        "T_head_tm1_head_t": T_head_tm1_head_t,
        # transformation from hands on previous frame to hands on current frame
        "T_hands_tm1_hands_t": T_hands_tm1_hands_t,
        # SMPL joints in world coordinates
        "smpl_joints_world": joints_th[1:-1],
        # Sparse features: R, w (omega - rotation velocity), XYZ, V
        "sparse_R_6d_global": sparse_R_6d_global,
        "sparse_w_global": sparse_w_global,
        "sparse_XYZ_global": sparse_XYZ_global[1:-1],
        "sparse_V_global": sparse_V_global,
        # Body and hand poses in 6d for supervision
        "body_pose_6d": body_pose_6d,
        "hand_pose_6d": hand_pose_6d,
    }


def extract_motion_features(smpl_model, smpl_params, device="cuda", fps=30):
    smpl_params_th = {k: torch.from_numpy(v).float() for k, v in smpl_params.items()}

    smpl_vertices_th, smpl_joints_th = infer_smpl(smpl_model, smpl_params_th, device)

    # Convert pose from pca if needed
    if smpl_model.use_pca:
        smpl_params_th["left_hand_pose"] = torch.einsum(
            "bi,ij->bj",
            [smpl_params_th["left_hand_pose"], smpl_model.left_hand_components],
        )
        smpl_params_th["right_hand_pose"] = torch.einsum(
            "bi,ij->bj",
            [smpl_params_th["right_hand_pose"], smpl_model.right_hand_components],
        )

    smpl_params_th = {k: v.cpu() for k, v in smpl_params_th.items()}

    motion_features = smpl_joints_to_motion_features(
        smpl_joints_th, smpl_model.parents.long(), smpl_params_th, fps=fps
    )

    # Crop to align with motion features
    motion_features["sbj_vertices"] = smpl_vertices_th.cpu()[1:-1]

    return motion_features


def get_object_features(
    obj_T: Dict[str, npt.NDArray],
    can_T: Dict[str, npt.NDArray],
    align_T: Dict[str, npt.NDArray],
    dataset: str,
):
    assert dataset in ["behave", "omomo"]

    # Prepare canonical object transform
    if dataset == "behave":
        R_can, t_can = can_T["R"], can_T["t"]
    elif dataset == "omomo":
        R_can, t_can, s_can = can_T["R"], can_T["t"], can_T["s"]
        t_can = s_can * t_can
    R_can_inv = np.linalg.inv(R_can)

    # Prepare alignment transform
    R_align, t_align = align_T["R"], align_T["t"]

    # Prepare object transform
    R_obj, t_obj = obj_T["R"], obj_T["t"]
    T = len(t_obj)

    # Obtain new transformation for the object taking alignment into account
    # Transformation is with respect to the new canonical position
    R_obj_new = np.matmul(
        R_align.reshape(1, 3, 3).repeat(T, 0),
        np.matmul(R_obj, R_can_inv.reshape(1, 3, 3).repeat(T, 0)),
    )

    t_obj_new = (
        -np.matmul(R_obj_new, t_can.reshape(1, 3, 1).repeat(T, 0))[:, :, 0]
        + np.dot(R_align, t_obj.T).T
        + t_align.reshape(1, 3)
    )

    return {
        "obj_R": R_obj_new,
        "obj_transl": t_obj_new,
    }


def canonicalize_object(obj_mesh):
    vertices = obj_mesh.vertices

    # 1. Center
    centroid = vertices.mean(axis=0)
    vertices_centered = vertices - centroid

    # 2. PCA for Alignment
    cov_matrix = np.cov(vertices_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 3. Define Z-up aligned basis
    # Assume Z = axis of greatest variance
    z_axis_idx = np.argmax(eigenvalues)
    v_z = eigenvectors[:, z_axis_idx]
    # Heuristic check for "up" direction
    if np.mean(vertices_centered @ v_z) < 0:
        v_z = -v_z
    remaining_indices = [i for i in range(3) if i != z_axis_idx]
    y_axis_idx = remaining_indices[np.argmax(eigenvalues[remaining_indices])]
    v_y = eigenvectors[:, y_axis_idx]

    # X = Y x Z (right-handed system)
    v_x = np.cross(v_y, v_z)
    v_x /= np.linalg.norm(v_x)
    v_y = np.cross(v_z, v_x)
    v_y /= np.linalg.norm(v_y)

    # 4. Construct Rotation Matrix R
    R = np.array([v_x, v_y, v_z])
    if np.linalg.det(R) < 0:
        R[0, :] = -R[0, :]  # Flip X axis if left-handed

    # 5. Calculate final translation t
    t = -R @ centroid

    # 6. Apply transformation
    obj_mesh_canonical = obj_mesh.copy()
    obj_mesh_canonical.vertices = np.dot(R, obj_mesh_canonical.vertices.T).T + t

    return obj_mesh_canonical, R, t


def transform_mesh(obj_mesh: trimesh.Trimesh, R, t) -> trimesh.Trimesh:
    """Applies v -> R @ v + t (the convention of canonicalize_object)."""
    obj_mesh_canonical = obj_mesh.copy()
    obj_mesh_canonical.vertices = np.dot(R, obj_mesh_canonical.vertices.T).T + t
    return obj_mesh_canonical


def objects_assets_folder(cfg: PreprocessConfig, dataset: str) -> Path:
    return Path(cfg.common.objects_assets_folder) / dataset


def load_canonical_transforms(cfg: PreprocessConfig, dataset: str) -> Optional[Dict]:
    """Canonical object transforms stored in the assets, None if not generated yet."""
    path = objects_assets_folder(cfg, dataset) / "canonical_object_transform.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as fp:
        return pkl.load(fp)


def sample_object_keypoints(
    object_mesh: trimesh.Trimesh, n_keypoints: int
) -> Dict[str, np.ndarray]:
    sampled_points = object_mesh.sample(n_keypoints)
    _, _, triangles_ids = trimesh.proximity.closest_point(object_mesh, sampled_points)
    barycentric = trimesh.triangles.points_to_barycentric(
        object_mesh.vertices[object_mesh.faces[triangles_ids]], sampled_points
    )
    return {
        "cartesian": sampled_points,
        "barycentric": barycentric,
        "triangles_ids": triangles_ids,
    }


def export_canonical_objects(
    cfg: PreprocessConfig,
    dataset: str,
    object_name2mesh: Dict[str, trimesh.Trimesh],
    canonical_transform: Dict[str, Any],
) -> None:
    """Writes the canonical meshes to <datasets_folder>/<dataset> and copies
    the canonical transforms, keypoints and pointnext features from the
    assets. Transforms and keypoints missing in the assets are generated once
    and added there, so later runs reproduce the same objects."""
    assets_folder = objects_assets_folder(cfg, dataset)
    target_folder = Path(cfg.env.datasets_folder) / dataset
    for folder in [
        assets_folder / "object_keypoints",
        target_folder / "object_meshes",
        target_folder / "object_keypoints",
    ]:
        folder.mkdir(exist_ok=True, parents=True)

    transform_asset = assets_folder / "canonical_object_transform.pkl"
    if not transform_asset.is_file():
        print(f"[{dataset}] storing canonical transforms in {transform_asset}")
        with transform_asset.open("wb") as fp:
            pkl.dump(canonical_transform, fp)
    shutil.copyfile(transform_asset, target_folder / "canonical_object_transform.pkl")

    for object_name, object_mesh in object_name2mesh.items():
        _ = object_mesh.export(str(target_folder / "object_meshes" / f"{object_name}.ply"))

        keypoints_asset = assets_folder / "object_keypoints" / f"{object_name}.npz"
        if keypoints_asset.is_file():
            # stored keypoints must correspond to the regenerated mesh: exact
            # barycentric reconstruction when they were sampled on this very
            # mesh, otherwise (keypoints inherited from a different
            # tessellation of the same object, e.g. the decimated OMOMO
            # meshes) they must lie on its surface up to a tolerance
            keypoints = np.load(keypoints_asset)
            deviation = None
            triangles_ids = keypoints["triangles_ids"]
            if triangles_ids.max() < len(object_mesh.faces):
                reconstructed = trimesh.triangles.barycentric_to_points(
                    object_mesh.vertices[object_mesh.faces[triangles_ids]],
                    keypoints["barycentric"],
                )
                deviation = np.abs(reconstructed - keypoints["cartesian"]).max()
            if deviation is None or deviation > 1e-4:
                _, surface_dist, _ = trimesh.proximity.closest_point(
                    object_mesh, keypoints["cartesian"]
                )
                if surface_dist.max() > cfg.common.keypoints_surface_tolerance:
                    raise RuntimeError(
                        f"[{dataset}] keypoints of {object_name} in {keypoints_asset} "
                        f"do not match the regenerated mesh (max surface distance "
                        f"{surface_dist.max():.2e})"
                    )
        else:
            print(f"[{dataset}] sampling keypoints for {object_name} -> {keypoints_asset}")
            np.savez(
                keypoints_asset,
                **sample_object_keypoints(object_mesh, cfg.common.n_keypoints),
            )
        shutil.copyfile(
            keypoints_asset, target_folder / "object_keypoints" / f"{object_name}.npz"
        )

    pointnext_asset = assets_folder / "object_pointnext.pkl"
    if pointnext_asset.is_file():
        with pointnext_asset.open("rb") as fp:
            missing = set(object_name2mesh) - set(pkl.load(fp))
        if missing:
            print(f"WARNING: [{dataset}] no pointnext features for {sorted(missing)}")
        shutil.copyfile(pointnext_asset, target_folder / "object_pointnext.pkl")
    else:
        print(
            f"WARNING: [{dataset}] {pointnext_asset} missing; pointnext features are "
            "generated outside this repo"
        )


def extract_segments(mask, length_thr=60, gap_thr=5):
    segments = []
    first_contact, in_segment, gap_length = -1, False, 0
    for i, contact in enumerate(mask):
        if contact:
            if not in_segment:
                in_segment = True
                first_contact = i
            gap_length = 0
        else:
            gap_length += 1

        if ((gap_length >= gap_thr) or (i == (len(mask) - 1))) and in_segment:
            segment_length = i - gap_length - first_contact + 1
            if segment_length >= length_thr:
                segments.append((first_contact, segment_length))

            in_segment = False
            first_contact = -1

    return segments


def convert_segments_to_mask(mask, segments):
    segment_masks = []
    for segment in segments:
        start, length = segment
        segment_mask = np.zeros_like(mask, dtype=bool)
        segment_mask[start : start + length] = True
        segment_masks.append(segment_mask)
    return segment_masks


def export_sequence_to_hdf5(
    bm_params: Dict[str, npt.NDArray],
    motion_features: Dict[str, npt.NDArray],
    object_features: Dict[str, npt.NDArray],
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    with h5py.File(output_path, "w") as hdf5_file:
        # Save metadata
        for k, v in metadata.items():
            if isinstance(v, str):
                hdf5_file.attrs[k] = v.encode("utf-8")
            else:
                hdf5_file.attrs[k] = v

        # Save body model parameters
        for k, v in bm_params.items():
            hdf5_file.create_dataset(k, data=v, chunks=True)

        # Save motion features
        for k, v in motion_features.items():
            hdf5_file.create_dataset(k, data=v, chunks=True)

        # Save object features
        if len(object_features) > 0:
            for k, v in object_features.items():
                hdf5_file.create_dataset(k, data=v, chunks=True)

    return None


def canonicalize_Rt(
    R: torch.Tensor, t: torch.Tensor, T_can: SE3, format="6d"
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert format in ["6d", "matrix", "aa"]

    if format == "6d":
        R = rotation_6d_to_matrix(R)
    elif format == "aa":
        R = aa_to_matrix(R)

    T = SE3.from_rotation_and_translation(SO3.from_matrix(R), t)
    T = T_can @ T
    R = T.rotation().as_matrix()
    t = T.translation()

    if format == "6d":
        R = matrix_to_rotation_6d(R)
    elif format == "aa":
        R = matrix_to_aa(R)

    return R, t


@overload
def get_reset_Z_rotation(
    R: npt.NDArray, eps_dot: float = 1e-4, eps_norm: float = 1e-6
) -> npt.NDArray: ...


@overload
def get_reset_Z_rotation(
    R: torch.Tensor, eps_dot: float = 1e-4, eps_norm: float = 1e-6
) -> torch.Tensor: ...


def get_reset_Z_rotation(
    R: Union[npt.NDArray, torch.Tensor], eps_dot: float = 1e-4, eps_norm: float = 1e-6
) -> Union[npt.NDArray, torch.Tensor]:
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
        is_numpy = False
    else:
        is_numpy = True

    # Handle both single matrix and batch of matrices
    if R.ndim == 2:
        # Single matrix case: add batch dimension
        R = R[np.newaxis, :, :]
        is_single = True
    else:
        is_single = False
    batch_size = R.shape[0]

    # Extract the Z-axis direction after rotation for all matrices
    z_rotated = R[:, :, 2]  # Shape: (B, 3)
    z_original = np.array([0, 0, 1], dtype=np.float32)

    # Compute dot product to check alignment for each matrix
    dot_z = np.dot(z_rotated, z_original)  # Shape: (B,)

    # Initialize output array
    twist_angles = np.zeros(batch_size, dtype=np.float32)

    # Case 1: Z-axis nearly unchanged (pure twist around Z)
    mask_aligned = dot_z > (1 - eps_dot)
    if np.any(mask_aligned):
        x_rotated = R[mask_aligned, :, 0]  # Shape: (N1, 3)
        twist_angles[mask_aligned] = np.arctan2(x_rotated[:, 1], x_rotated[:, 0])

    def twist_from_swing(tangents: npt.NDArray, R_sub: npt.NDArray) -> npt.NDArray:
        # Extract twist: R = R_swing @ R_twist => R_twist = R_swing^T @ R
        R_swing = (
            SO3.exp(torch.from_numpy(tangents.astype(np.float32))).as_matrix().numpy()
        )
        R_twist = np.einsum("nji,njk->nik", R_swing, R_sub)
        return np.arctan2(R_twist[:, 1, 0], R_twist[:, 0, 0])

    # Case 2: Z-axis nearly flipped (180° - critical case)
    mask_flipped = dot_z < -(1 - eps_dot)
    if np.any(mask_flipped):
        R_flipped = R[mask_flipped]
        xy_component = R_flipped[:, :2, 0]  # xy of the rotated x-axis, (N2, 2)
        xy_norm = np.linalg.norm(xy_component, axis=1)

        swing_axes = np.zeros((len(R_flipped), 3), dtype=np.float32)
        safe = xy_norm > eps_norm
        swing_axes[~safe, 0] = 1.0
        swing_axes[safe, 0] = -xy_component[safe, 1] / xy_norm[safe]
        swing_axes[safe, 1] = xy_component[safe, 0] / xy_norm[safe]

        twist_angles[mask_flipped] = twist_from_swing(swing_axes * np.pi, R_flipped)

    # Case 3: General case (no singularity)
    mask_general = ~(mask_aligned | mask_flipped)
    if np.any(mask_general):
        R_general = R[mask_general]

        # Find the swing rotation (no twist component)
        swing_axes = np.cross(z_original[None, :], z_rotated[mask_general])
        swing_axes = swing_axes / np.linalg.norm(swing_axes, axis=1, keepdims=True)
        swing_angles = np.arccos(np.clip(dot_z[mask_general], -1, 1))

        twist_angles[mask_general] = twist_from_swing(
            swing_axes * swing_angles[:, None], R_general
        )

    # Convert twist angles to rotation matrices
    R_z = SO3.from_z_radians(-torch.from_numpy(twist_angles)).as_matrix()

    if is_single:
        # Remove batch dimension for single matrix input
        R_z = R_z[0]

    if is_numpy:
        R_z = R_z.numpy()

    return R_z
