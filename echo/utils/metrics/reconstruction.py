from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch

from echo.configs import ProjectConfig
from echo.data import dataset_name_to_config
from echo.data.dataset import MotionDataset
from echo.model.mesh_model import MeshModel
from echo.utils.contacts import compute_env_contactness, compute_hoi_contactness
from echo.utils.rotation import matrix_to_quat


@torch.no_grad()
def contacts_worker(sbj_verts, obj_verts, contact_threshold):
    with torch.no_grad():
        contacts = torch.cdist(
            sbj_verts,
            obj_verts,
        )
        contacts = contacts.min(dim=-1).values.cpu().numpy()
        contacts_mask = contacts <= contact_threshold
    return contacts_mask


def get_mesh_based_contacts(sbj, obj, contact_threshold, batch=1000):
    T = sbj.shape[0]

    input_data = [
        {
            "obj_verts": obj[t : min(t + batch, T)],
            "sbj_verts": sbj[t : min(t + batch, T)],
            "contact_threshold": contact_threshold,
        }
        for t in range(0, T, batch)
    ]

    contacts_mask = [contacts_worker(**input_data[t]) for t in range(len(input_data))]
    contacts_mask = np.concatenate(contacts_mask, axis=0)
    return contacts_mask


def get_mpjpe(pred_joints, gt_joints):
    mpjpe = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=-1))
    return mpjpe.mean(tuple(range(1, mpjpe.ndim))).sum() * 100


def get_mpjve(pred_joints, gt_joints):
    pred_velocity = pred_joints[:, 1:] - pred_joints[:, :-1]
    gt_velocity = gt_joints[:, 1:] - gt_joints[:, :-1]

    mpjve = np.sqrt(np.sum((gt_velocity - pred_velocity) ** 2, axis=-1))
    return mpjve.mean(tuple(range(1, mpjve.ndim))).sum() * 100


def get_foot_skate(pred_joints):
    # From EgoAllo
    # pred_joints are TxJx3
    feet_indices = torch.tensor([7, 8, 10, 11], dtype=torch.long)

    # normalize height using foot joint
    _pred_joints = torch.tensor(pred_joints)
    _pred_joints[:, :, 2] -= torch.min(_pred_joints[:, :, 2])

    # compute difference in XY plane (Z is up)
    feet_position = _pred_joints[:, feet_indices]
    feet_position_diff = feet_position[1:, :, [0, 1]] - feet_position[:-1, :, [0, 1]]
    feet_positions_diff_norm = torch.sum(torch.abs(feet_position_diff), dim=-1)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        dtype=torch.float32,
    )

    # Threshold height
    feet_positions_diff_norm = feet_positions_diff_norm * (
        feet_position[1:, :, 2] < H_thresh
    )
    skating = torch.sum(
        torch.sum(
            feet_positions_diff_norm * (2 - 2 ** (feet_position[1:, :, 2] / H_thresh)),
            dim=-1,
        ),
        dim=-1,
    )

    return skating.item()


def get_foot_contact(pred_joints):
    # assumes y-axis is up
    # from EgoAllo
    # pred_joints are TxJx3
    feet_indices = torch.tensor([7, 8, 10, 11], dtype=torch.long)
    T = pred_joints.shape[0]

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.1, 0.1, 0.05, 0.05],
        dtype=torch.float32,
    )

    feet_positions = torch.tensor(pred_joints)[:, feet_indices]
    any_contact = torch.any(
        torch.any(feet_positions[:, :, 2] < H_thresh, dim=-1), dim=-1
    ).to(torch.float32)

    return any_contact.item()


def get_contact_metrics(pred_contact, gt_contact):
    pred_contact = pred_contact > 0.25
    gt_contact = gt_contact > 0.5
    pred_flat = pred_contact.flatten()
    gt_flat = gt_contact.flatten()

    tp = np.sum(np.logical_and(pred_flat, gt_flat))
    tn = np.sum(np.logical_and(~pred_flat, ~gt_flat))
    fp = np.sum(np.logical_and(pred_flat, ~gt_flat))
    fn = np.sum(np.logical_and(~pred_flat, gt_flat))

    total = tp + tn + fp + fn
    if total == 0:
        return {
            "acc": 1.0,
            "prec": 1.0,
            "rec": 1.0,
            "f1": 1.0,
        }

    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "acc": 100 * accuracy,
        "prec": 100 * precision,
        "rec": 100 * recall,
        "f1": f1,
    }


def get_sbj_metrics(config: ProjectConfig, samples_folder: Path, dataset_name: str):
    dataset_config, kwargs = dataset_name_to_config(dataset_name, config)

    dataset = MotionDataset(
        name=dataset_config.name,
        root=Path(dataset_config.root),
        split="test",
        assets_folder=Path(config.env.assets_folder),
        conditioning_config=config.model_conditioning,
        model_config=config.model,
        split_file=dataset_config.test_split_file,
        num_betas=config.model_mesh.num_betas,
        fps=dataset_config.fps_eval,
        slice_strategy="deterministic",
        **kwargs,
    )

    mpjpe, mpjve, skate, contact, skate_gt, contact_gt = [], [], [], [], [], []
    env_contact_metrics = defaultdict(list)
    for sequence, test_h5py in zip(dataset.sequences, dataset.h5datasets):
        if not (samples_folder / f"{sequence}.hdf5").exists():
            continue

        with h5py.File(samples_folder / f"{sequence}.hdf5", "r") as sample_h5py:
            T = test_h5py.attrs["T"]

            gt_joints = test_h5py["smpl_joints_world"][:]
            sampled_joints = sample_h5py["smpl_joints_world"][:]

            N_J = 22  # use only body joints
            sampled_joints = sampled_joints[:T, :N_J]
            gt_joints = gt_joints[:T, :N_J]

            mpjpe.append(get_mpjpe(sampled_joints, gt_joints) / T)
            mpjve.append(get_mpjve(sampled_joints, gt_joints) / T)
            skate.append(get_foot_skate(sampled_joints))
            skate_gt.append(get_foot_skate(gt_joints))
            contact.append(get_foot_contact(sampled_joints))
            contact_gt.append(get_foot_contact(gt_joints))

            if "diff_contacts" in sample_h5py and "env_contacts" in test_h5py:
                if (
                    sample_h5py["diff_contacts"][:].shape[1]
                    != config.model.dim_contacts_hoi + config.model.dim_contacts_env
                ):
                    continue
                diff_contacts = np.asarray(sample_h5py["diff_contacts"][:])[:T]
                dim_hoi = config.model.dim_contacts_hoi
                pred_env_contacts = diff_contacts[:, dim_hoi:]

                gt_env_contacts_raw = test_h5py["env_contacts"][:][:T]
                gt_contacts = compute_env_contactness(
                    torch.from_numpy(gt_env_contacts_raw)
                ).numpy()

                metrics = get_contact_metrics(pred_env_contacts, gt_contacts)
                for k, v in metrics.items():
                    env_contact_metrics[k].append(v)

    mpjpe = np.stack(mpjpe)
    mpjve = np.stack(mpjve)
    skate = np.stack(skate)
    skate_gt = np.stack(skate_gt)
    contact = np.stack(contact)
    contact_gt = np.stack(contact_gt)
    for k, v in env_contact_metrics.items():
        env_contact_metrics[k] = np.stack(v)

    return mpjpe, mpjve, skate, skate_gt, contact, contact_gt, env_contact_metrics


def get_mpjpe_object(pred_keypoints, gt_keypoints):
    # from InterDiff
    # the average l2 distance between the predicted and
    # ground truth keypoints of the object
    mpjpe = np.sqrt(np.sum((pred_keypoints - gt_keypoints) ** 2, axis=-1))
    return mpjpe.mean(tuple(range(1, mpjpe.ndim))).sum() * 100


def get_rot_diff(pred_rot, gt_rot):
    theta = np.clip(
        (np.trace(np.matmul(pred_rot, gt_rot.transpose(0, 2, 1)), axis1=1, axis2=2) - 1)
        / 2,
        a_min=-1.0,
        a_max=1.0,
    )
    rot_diff_rad = np.arccos(theta) % (2 * np.pi)
    rot_diff = np.rad2deg(rot_diff_rad)

    return rot_diff.mean()


def get_quat_diff(pred_quat, gt_quat):
    # from InterDiff
    # the average l1 distance between the predicted and
    # ground truth quaternions of the object
    return np.linalg.norm(pred_quat - gt_quat, axis=1, ord=1).mean()


def get_obj_metrics(config: ProjectConfig, samples_folder: Path, dataset_name: str):
    dataset_config, kwargs = dataset_name_to_config(dataset_name, config)

    dataset = MotionDataset(
        name=dataset_config.name,
        root=Path(dataset_config.root),
        split="test",
        assets_folder=Path(config.env.assets_folder),
        conditioning_config=config.model_conditioning,
        model_config=config.model,
        split_file=dataset_config.test_split_file,
        num_betas=config.model_mesh.num_betas,
        fps=dataset_config.fps_eval,
        slice_strategy="deterministic",
        **kwargs,
    )

    # pad keypoints
    obj_keypoints = dataset.obj_keypoints
    if min(obj_keypoints.keys()) > 0:
        min_class_id = min(obj_keypoints.keys())
        for class_id in range(min_class_id):
            obj_keypoints[class_id] = deepcopy(obj_keypoints[min_class_id])

    mesh_model: MeshModel = MeshModel(
        model_type=config.model_mesh.type,
        env=config.env,
        num_betas=config.model_mesh.num_betas,
        include_object=True,
        device="cuda",
        inference_chunk_size=config.model_mesh.inference_chunk_size,
        batch_size=config.model_mesh.inference_chunk_size,
        canonical_obj_keypoints=obj_keypoints,
        infer_joints_only=False,
    )

    mpjpe_obj, obj_c, obj_rot_diff, obj_quat_diff = [], [], [], []
    diff_contact_metrics = defaultdict(list)
    mesh_contact_metrics = defaultdict(list)
    for sequence, test_h5py in zip(dataset.sequences, dataset.h5datasets):
        if not (samples_folder / f"{sequence}.hdf5").exists():
            continue

        with h5py.File(samples_folder / f"{sequence}.hdf5", "r") as sample_h5py:
            T = test_h5py.attrs["T"]

            object_name = sample_h5py.attrs["object_name"]
            object_id = dataset.obj2classid[object_name]

            can_obj_keypoints = dataset.obj_keypoints[object_id]["cartesian"]
            can_obj_keypoints = torch.tensor(can_obj_keypoints)
            can_obj_keypoints = can_obj_keypoints.unsqueeze(0).repeat_interleave(T, 0)
            can_obj_keypoints = can_obj_keypoints.numpy()

            sampled_obj_global_Rt_6d = sample_h5py["obj_Rt_6d"][:]
            sampled_obj_global_R = sample_h5py["obj_R"][:]
            sampled_obj_global_q = matrix_to_quat(
                torch.tensor(sampled_obj_global_R)
            ).numpy()
            # pose object keypoints
            sampled_obj_keypoints = (
                np.matmul(
                    sampled_obj_global_R, np.copy(can_obj_keypoints).transpose(0, 2, 1)
                ).transpose(0, 2, 1)
                + sampled_obj_global_Rt_6d[:, None, 6:]
            )

            gt_obj_global_t = test_h5py["obj_transl"][:T]
            gt_obj_global_R = test_h5py["obj_R"][:T]
            gt_obj_global_q = matrix_to_quat(torch.tensor(gt_obj_global_R)).numpy()
            # pose object keypoints
            gt_obj_keypoints = (
                np.matmul(
                    gt_obj_global_R, np.copy(can_obj_keypoints).transpose(0, 2, 1)
                ).transpose(0, 2, 1)
                + gt_obj_global_t[:, None]
            )

            mpjpe_obj.append(
                get_mpjpe_object(sampled_obj_keypoints, gt_obj_keypoints) / T
            )

            obj_c.append(
                np.linalg.norm(
                    sampled_obj_global_Rt_6d[:, 6:] - gt_obj_global_t, axis=-1
                ).sum()
                * 100
                / T
            )

            obj_rot_diff.append(get_rot_diff(sampled_obj_global_R, gt_obj_global_R))

            obj_quat_diff.append(get_quat_diff(sampled_obj_global_q, gt_obj_global_q))

            if "diff_contacts" in sample_h5py and "hoi_contacts" in test_h5py:
                if (
                    sample_h5py["diff_contacts"][:].shape[1]
                    != config.model.dim_contacts_hoi + config.model.dim_contacts_env
                ):
                    continue

                diff_contacts = np.asarray(sample_h5py["diff_contacts"][:])
                dim_hoi = config.model.dim_contacts_hoi
                pred_hoi_contacts = diff_contacts[:, :dim_hoi]

                gt_hoi_contacts_raw = test_h5py["hoi_contacts"][:T]
                gt_contacts = compute_hoi_contactness(
                    torch.from_numpy(gt_hoi_contacts_raw)
                ).numpy()

                metrics = get_contact_metrics(pred_hoi_contacts, gt_contacts)
                for k, v in metrics.items():
                    diff_contact_metrics[k].append(v)

            # mesh based contact
            gt_mesh_output = mesh_model.get_meshes_from_h5(
                test_h5py, object_id, with_skin=True
            )
            gt_sbj = gt_mesh_output.sbj_vertices.squeeze(0)[
                :, mesh_model.smpl_downsample_indices
            ]
            gt_obj = gt_mesh_output.obj_keypoints.squeeze(0)
            gt_mesh_contacts = get_mesh_based_contacts(
                gt_sbj, gt_obj, contact_threshold=0.1
            )

            pred_mesh_output = mesh_model.get_meshes_from_h5(
                sample_h5py, object_id, with_skin=True
            )
            pred_sbj = pred_mesh_output.sbj_vertices.squeeze(0)[
                :, mesh_model.smpl_downsample_indices
            ]
            pred_obj = pred_mesh_output.obj_keypoints.squeeze(0)
            pred_mesh_contacts = get_mesh_based_contacts(
                pred_sbj, pred_obj, contact_threshold=0.1
            )

            metrics = get_contact_metrics(pred_mesh_contacts, gt_mesh_contacts)
            for k, v in metrics.items():
                mesh_contact_metrics[k].append(v)

    mpjpe_obj = np.stack(mpjpe_obj)
    obj_c = np.stack(obj_c)
    obj_rot_diff = np.stack(obj_rot_diff)
    obj_quat_diff = np.stack(obj_quat_diff)
    for k, v in diff_contact_metrics.items():
        diff_contact_metrics[k] = np.stack(v)

    return (
        mpjpe_obj,
        obj_c,
        obj_rot_diff,
        obj_quat_diff,
        diff_contact_metrics,
        mesh_contact_metrics,
    )
