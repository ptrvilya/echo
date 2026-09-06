import logging
from pathlib import Path
from typing import Dict, Final, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from sklearn.cluster import DBSCAN
from typing_extensions import overload

from echo.configs import PreprocessConfig
from echo.utils.rotation import aa_to_matrix, matrix_to_aa

logging = logging.getLogger(__name__)

# Constants
SMPL_JOINT_TO_ID: Final[Dict[str, int]] = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}
ENV_CONTACT_JOINTS: Final[List[str]] = [
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
    "left_wrist",
    "right_wrist",
]
ENV_CONTACT_JOINT_TO_IDX: Final[Dict[str, int]] = {
    name: i for i, name in enumerate(ENV_CONTACT_JOINTS)
}
EYEJAW_START: Final[int] = 22
EYEJAW_END: Final[int] = 25
ENV_CONTACT_VERTICES: Final[List[int]] = [
    3677,  # left_knee
    6348,  # right_knee
    5881,  # left_ankle
    8575,  # right_ankle
    5912,  # left_foot
    8606,  # right_foot
    4812,  # left_wrist
    7548,  # right_wrist
]


def load_neutral_beta_conversion(
    config: PreprocessConfig, gender: str, num_betas: int = 16
) -> Tuple[npt.NDArray, npt.NDArray]:
    assert gender in ["female", "male"]
    assert num_betas in [10, 16]

    if num_betas == 16:
        suffix = "_n_comps_16"
    else:
        suffix = ""
    data = np.load(
        Path(config.env.assets_folder)
        / f"gender_conversion{suffix}"
        / f"smplx_{gender}_to_neutral.npz"
    )
    return data["A"], data["b"]


def convert_to_neutral_beta(
    betas: npt.NDArray, A: npt.NDArray, b: npt.NDArray
) -> npt.NDArray:
    """
    :param beta (*, B)
    :param A (B, B)
    :param b (B)
    beta_neutral = A @ beta_gender + b
    """
    *dims, B = betas.shape

    A = A.reshape((*(1,) * len(dims), B, B))
    b = b.reshape((*(1,) * len(dims), B))
    return np.einsum("...ij,...j->...i", A, betas) + b


@overload
def vposer2smplx_keys(bm_params: Dict[str, npt.NDArray]) -> Dict[str, npt.NDArray]: ...


@overload
def vposer2smplx_keys(
    bm_params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]: ...


def vposer2smplx_keys(
    bm_params: Union[Dict[str, torch.Tensor], Dict[str, npt.NDArray]],
) -> Union[Dict[str, torch.Tensor], Dict[str, npt.NDArray]]:
    key_mapping = {
        "root_orient": "global_orient",
        "pose_body": "body_pose",
        # "pose_hand" and "pose_eye" are processed separately
        "pose_jaw": "jaw_pose",
        "trans": "transl",
        "betas": "betas",
    }

    keys = list(bm_params.keys())
    for key in keys:
        param = bm_params.pop(key)
        if key == "pose_hand":
            bm_params["left_hand_pose"] = param[:, :45]
            bm_params["right_hand_pose"] = param[:, 45:]
        elif key == "pose_eye":
            bm_params["leye_pose"] = param[:, :3]
            bm_params["reye_pose"] = param[:, 3:]
        else:
            bm_params[key_mapping[key]] = param
    return bm_params


def local2global_pose(
    local_pose: torch.Tensor, parent_ids: torch.Tensor
) -> torch.Tensor:
    """
    Convert local pose to global pose by multiplying with parent transforms.
    """
    if local_pose.ndim == 4:
        B = local_pose.shape[0]
    else:
        B = 1

    local_pose = local_pose.view(B, -1, 3, 3)
    global_pose = local_pose.detach().clone()

    for joint_id in range(len(parent_ids)):
        parent_id = parent_ids[joint_id]
        if parent_id >= 0:
            global_pose[:, joint_id] = torch.matmul(
                global_pose[:, parent_id], global_pose[:, joint_id]
            )

    if B == 1:
        global_pose = global_pose.squeeze(0)

    return global_pose


def compute_height_and_contacts(
    body_joint_seq,
    fps,
    floor_vel_thresh=0.005,
    floor_height_offset=0.01,
    contact_vel_thresh=0.005,  # 0.015
    contact_toe_height_thresh=0.04,  # if static toe above this height
    contact_ankle_height_thresh=0.08,
    terrain_height_thresh=0.04,
    root_height_thresh=0.04,
    cluster_size_thresh=0.25,
    discard_terrain_seqs=False,  # throw away person steps onto objects (determined by a heuristic)
):
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py

    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    """
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINT_TO_ID["pelvis"], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINT_TO_ID["left_foot"], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINT_TO_ID["right_foot"], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < floor_vel_thresh]
    left_static_inds = all_inds[left_toe_vel < floor_vel_thresh]
    right_static_foot_heights = right_toe_heights[right_toe_vel < floor_vel_thresh]
    right_static_inds = all_inds[right_toe_vel < floor_vel_thresh]

    all_static_foot_heights = np.append(
        left_static_foot_heights, right_static_foot_heights
    )
    all_static_inds = np.append(left_static_inds, right_static_inds)

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(
            all_static_foot_heights.reshape(-1, 1)
        )
        all_labels = np.unique(clustering.labels_)

        min_median = min_root_median = float("inf")
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(
                all_static_inds[clustering.labels_ == cur_label]
            )  # inds in the original sequence that correspond to this cluster

            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median
        offset_floor_height = (
            floor_height - floor_height_offset
        )  # toe joint is actually inside foot mesh a bit

        if discard_terrain_seqs:
            for cluster_root_height, cluster_height, cluster_size in zip(
                cluster_root_heights, cluster_heights, cluster_sizes
            ):
                root_above_thresh = cluster_root_height > (
                    min_root_median + root_height_thresh
                )
                toe_above_thresh = cluster_height > (min_median + terrain_height_thresh)
                cluster_size_above_thresh = cluster_size > int(
                    cluster_size_thresh * fps
                )
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    logging.warning("DISCARDING sequence based on terrain interaction!")
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINT_TO_ID["left_ankle"], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINT_TO_ID["right_ankle"], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights = left_toe_heights - floor_height
    right_toe_heights = right_toe_heights - floor_height

    # Compute velocity and height for flexible processing in the dataloader
    contacts = np.zeros((num_frames, len(ENV_CONTACT_JOINTS), 2))
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["left_ankle"]] = np.stack(
        [left_heel_vel, left_heel_heights], axis=1
    )
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["left_foot"]] = np.stack(
        [left_toe_vel, left_toe_heights], axis=1
    )
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["right_ankle"]] = np.stack(
        [right_heel_vel, right_heel_heights], axis=1
    )
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["right_foot"]] = np.stack(
        [right_toe_vel, right_toe_heights], axis=1
    )

    # hand contacts
    left_hand_contact = calculate_joint_contact(
        body_joint_seq,
        "left_wrist",
        floor_height,
    )
    right_hand_contact = calculate_joint_contact(
        body_joint_seq,
        "right_wrist",
        floor_height,
    )
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["left_wrist"]] = left_hand_contact
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["right_wrist"]] = right_hand_contact

    # knee contacts
    left_knee_contact = calculate_joint_contact(
        body_joint_seq,
        "left_knee",
        floor_height,
    )
    right_knee_contact = calculate_joint_contact(
        body_joint_seq,
        "right_knee",
        floor_height,
    )
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["left_knee"]] = left_knee_contact
    contacts[:, ENV_CONTACT_JOINT_TO_IDX["right_knee"]] = right_knee_contact

    return offset_floor_height, contacts, discard_seq


def calculate_joint_contact(body_joint_seq, joint_name, floor_height):
    """
    Adapted from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py
    """
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINT_TO_ID[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])

    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height

    return np.stack([joint_vel, joint_heights], axis=1)


def infer_smpl(smpl_model, smpl_params: Dict[str, torch.Tensor], device="cuda"):
    smpl_params = {k: v.to(device) for k, v in smpl_params.items()}

    B = smpl_params["betas"].shape[0]
    for k in smpl_params.keys():
        if k != "betas":
            smpl_params[k] = smpl_params[k].reshape(B, -1)

    if "expression" not in smpl_params:
        smpl_params["expression"] = torch.zeros((B, 10), dtype=torch.float).to(device)

    if "jaw_pose" not in smpl_params:
        smpl_params["jaw_pose"] = torch.zeros((B, 3), dtype=torch.float).to(device)

    if "leye_pose" not in smpl_params:
        smpl_params["leye_pose"] = torch.zeros((B, 3), dtype=torch.float).to(device)

    if "reye_pose" not in smpl_params:
        smpl_params["reye_pose"] = torch.zeros((B, 3), dtype=torch.float).to(device)

    with torch.no_grad():
        smplx_output = smpl_model(**smpl_params)
    smpl_joints_th = smplx_output["joints"].cpu()
    smpl_vertices_th = smplx_output["vertices"].cpu()

    return smpl_vertices_th, smpl_joints_th


def get_alignment_transform_and_align_smpl(
    R_align: npt.NDArray, smpl_params: Dict[str, npt.NDArray], smpl_model, device="cuda"
):
    # Calculate updated global rotation
    T = smpl_params["global_orient"].shape[0]
    smpl_sequence_global_R = aa_to_matrix(smpl_params["global_orient"])
    smpl_sequence_global_R = np.matmul(
        R_align[None].repeat(T, 0), smpl_sequence_global_R
    )
    smpl_params["global_orient"] = matrix_to_aa(smpl_sequence_global_R)

    # Infer SMPL
    smpl_params_th = {
        key: torch.from_numpy(value).float().to(device)
        for key, value in smpl_params.items()
    }
    smpl_vertices_th, smpl_joints_th = infer_smpl(smpl_model, smpl_params_th)
    smpl_vertices = smpl_vertices_th.cpu().numpy()
    smpl_joints = smpl_joints_th.cpu().numpy()

    # Re-compute translation taking rotation around the root into account
    # Info: https://github.com/vchoutas/smplx/tree/main?tab=readme-ov-file#modifying-the-global-pose-of-the-model
    # pelvis is not subject to global rotation
    pelvis = smpl_joints[:, 0, :] - smpl_params["transl"]
    smpl_transl_aligned = (
        np.matmul(R_align, (smpl_params["transl"] + pelvis)[..., None]).squeeze(-1)
        - pelvis
    )
    smpl_vertices_aligned = (
        smpl_vertices - smpl_params["transl"][:, None] + smpl_transl_aligned[:, None]
    )

    # Find translation to align the ground plane and start point
    transl_align = np.zeros(3)
    transl_align[2] = np.min(smpl_vertices_aligned[:, :, 2])
    transl_align[:2] = np.mean(smpl_vertices_aligned[0, :, :2], axis=0)
    transl_align *= -1
    smpl_params["transl"] = smpl_transl_aligned + transl_align.reshape(1, 3)

    return transl_align
