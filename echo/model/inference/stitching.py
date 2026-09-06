from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers import SchedulerMixin
from torch import Tensor

from echo.data import BatchData
from echo.model.inference.naive import InferenceModelNaive
from echo.model.output import EgoModelOutput
from echo.utils.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix
from echo.utils.transforms import SE3, SO3


class InferenceStitcher(InferenceModelNaive):
    """Assumes batch size == 1"""

    def __init__(
        self,
        window_size: int,
        num_inference_steps: int = 100,
        dim_contacts: int = 0,
        overlap_size: int = 64,
        num_joints: int = 21,
        object_pose="relative",
    ):
        super().__init__(
            window_size=window_size, num_inference_steps=num_inference_steps
        )
        self.overlap_size = overlap_size
        self.window_size = window_size
        self.num_joints = num_joints
        self.dim_contacts = dim_contacts
        self.num_inference_steps = num_inference_steps
        self.object_pose = object_pose

        # accumulate in rotation matrices for averaging
        # after averaging project back to rotation matrix via svd
        self._reset()

    def _reset(self) -> None:
        # main inference data
        self.sequence_body_pose_mat = torch.empty(0)
        self.sequence_object_pose_R_mat = torch.empty(0)
        self.sequence_object_pose_t = torch.empty(0)
        self.sequence_contacts = torch.empty(0)

        self.sequence_overlap_weights = torch.empty(0)
        self.overlap_weights = torch.empty(0)

        # placeholder to accumulate output data and conditioning
        self.sequence_output = EgoModelOutput()

        return None

    def _initialize(self, sequence_len: int, device: torch.device) -> None:
        self.sequence_body_pose_mat = torch.zeros(
            (sequence_len, self.num_joints, 3, 3), dtype=torch.float32, device=device
        )
        self.sequence_object_pose_R_mat = torch.zeros(
            (sequence_len, 3, 3), dtype=torch.float32, device=device
        )
        self.sequence_object_pose_t = torch.zeros(
            (sequence_len, 3), dtype=torch.float32, device=device
        )
        self.sequence_contacts = torch.zeros(
            (sequence_len, self.dim_contacts), dtype=torch.float32, device=device
        )

        self.sequence_output = EgoModelOutput()

        self.sequence_overlap_weights = torch.zeros(
            (sequence_len, 1), dtype=torch.float32, device=device
        )
        # from egoallo
        self.overlap_weights = (
            torch.from_numpy(
                np.minimum(
                    # Make this shape /```\
                    self.overlap_size,
                    np.minimum(
                        # Make this shape: /
                        np.arange(1, self.window_size + 1),
                        # Make this shape: \
                        np.arange(1, self.window_size + 1)[::-1],
                    ),
                )
                / self.overlap_size,
            )
            .float()
            .view(-1, 1)
            .to(device)
        )

        return None

    @staticmethod
    def _convert_6d_to_matrix(pose_6d: Tensor) -> Tensor:
        window_length = pose_6d.shape[1]
        segment_pose_6d = pose_6d.reshape(window_length, -1, 6)
        segment_pose_mat = rotation_6d_to_matrix(segment_pose_6d)
        return segment_pose_mat

    @staticmethod
    def project_rotation_matrices(matrices: Tensor) -> Tensor:
        # Assumes matrices are T x N_J x 3 x 3
        T, N_J = matrices.shape[:2]
        matrices_flat = matrices.reshape(-1, 3, 3)

        # Project back to SO(3) using SVD
        U, S, Vh = torch.linalg.svd(matrices_flat)

        # Compute the projection: U @ V^T
        matrices_projected = torch.einsum("...ij,...jk->...ik", U, Vh)

        # Ensure proper rotation (det = 1)
        # If det < 0, we have a reflection; flip the sign of the last column of U
        det = torch.det(matrices_projected)
        correction = torch.where(det < 0, -1.0, 1.0).view(-1, 1)

        # Apply correction to last column of U
        U[:, :, -1] = U[:, :, -1] * correction

        # Recompute projection with corrected U
        matrices_projected = torch.einsum("...ij,...jk->...ik", U, Vh)

        return matrices_projected.reshape(T, N_J, 3, 3)

    def before_inference(
        self,
        noise: torch.Tensor,
        batch: BatchData,
        scheduler: SchedulerMixin,
        t_index: Optional[torch.Tensor] = None,
        condition_frame: Optional[torch.Tensor] = None,
        condition_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # initialize
        # assumes batch size == 1
        if self.sequence_body_pose_mat.numel() == 0:
            total_sequence_len = batch.total_sequence_len[0].item()
            device = batch.body_pose_6d.device
            self._initialize(total_sequence_len, device=device)

        return noise.clone(), condition_frame, condition_global, t_index

    def before_step(
        self,
        noise: torch.Tensor,
        batch: BatchData,
        t_index: Optional[torch.Tensor] = None,
        condition_frame: Optional[torch.Tensor] = None,
        condition_global: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return noise, condition_frame, condition_global, t_index

    def on_inference(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        scheduler: SchedulerMixin,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return pred

    def on_scheduler(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        scheduler: SchedulerMixin,
    ) -> torch.Tensor:
        return x_t

    def update_timestep(
        self,
        t_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return t_index + 1

    def after_inference(
        self,
        batch: BatchData,
        x_t: torch.Tensor,
    ) -> EgoModelOutput:
        # unpack output
        output = EgoModelOutput().unpack(
            x_t,
            timesteps_motion=torch.zeros(1, x_t.shape[1]),
            batch=batch,
            object_pose=self.object_pose,
        )

        # Assumes batch_size == 1
        # Convert to rotation matrices
        segment_body_pose_mat = self._convert_6d_to_matrix(output.body_pose_6d)
        if output.T_world_obj is not None:
            T_world_obj = SE3(output.T_world_obj.reshape(-1, 7)).as_matrix()
            segment_object_pose_R_mat = T_world_obj[:, :3, :3]
            segment_object_pose_t = T_world_obj[:, :3, 3]

        # take the correct slice of the weights
        segment_length = batch.real_slice_len[0].item()
        segment_overlap_weights = self.overlap_weights[:segment_length]

        # Select correct part of the sequence
        t_start, t_end = batch.t_start.item(), batch.t_end.item()

        # Accumulated predictions and weights
        self.sequence_overlap_weights[t_start:t_end, :] += segment_overlap_weights
        self.sequence_body_pose_mat[t_start:t_end] += (
            segment_overlap_weights.view(-1, 1, 1, 1)
            * segment_body_pose_mat[:segment_length]
        )
        if output.T_world_obj is not None:
            self.sequence_object_pose_R_mat[t_start:t_end] += (
                segment_overlap_weights.view(-1, 1, 1)
                * segment_object_pose_R_mat[:segment_length]
            )
            self.sequence_object_pose_t[t_start:t_end] += (
                segment_overlap_weights.view(-1, 1)
                * segment_object_pose_t[:segment_length]
            )
        if output.diff_contacts is not None:
            self.sequence_contacts[t_start:t_end] += (
                segment_overlap_weights.view(-1, 1)
                * output.diff_contacts[0, :segment_length]
            )

        # Accumulate conditions and meta information
        output.body_pose_6d = None  # will be overwritten afterwards
        output.obj_Rt_6d = None  # will be overwritten afterwards
        output.T_world_obj = None  # will be overwritten afterwards
        output.diff_contacts = None  # will be overwritten afterwards
        if self.sequence_output.is_empty():
            self.sequence_output = output
        else:
            for k, v in output.items():
                if v is not None:
                    if k == "real_slice_len":
                        self.sequence_output[k] += segment_length - self.overlap_size
                    elif k == "obj_class_id":
                        continue
                    else:
                        self.sequence_output[k] = torch.cat(
                            [self.sequence_output[k], v[:, self.overlap_size :]],
                            dim=1,
                        )
        # Determine if we return the result or skip
        if batch.is_end_of_sequence[0].item():
            # copy
            body_pose_mat = self.sequence_body_pose_mat.clone()
            object_pose_R_mat = self.sequence_object_pose_R_mat.clone()
            object_pose_t = self.sequence_object_pose_t.clone()
            diff_contacts = self.sequence_contacts.clone()

            # Normalize
            body_pose_mat = body_pose_mat / self.sequence_overlap_weights.view(
                -1, 1, 1, 1
            )
            if object_pose_R_mat.numel() > 0:
                object_pose_R_mat = (
                    object_pose_R_mat / self.sequence_overlap_weights.view(-1, 1, 1)
                )
            if object_pose_t.numel() > 0:
                object_pose_t = object_pose_t / self.sequence_overlap_weights.view(
                    -1, 1
                )
            if diff_contacts.numel() > 0:
                diff_contacts = diff_contacts / self.sequence_overlap_weights.view(
                    -1, 1
                )

            # Project
            body_pose_mat = self.project_rotation_matrices(body_pose_mat)
            object_pose_R_mat = self.project_rotation_matrices(
                object_pose_R_mat.unsqueeze(1)
            ).squeeze(1)

            # Construct final output
            T = body_pose_mat.shape[0]
            sequence_output = self.sequence_output
            sequence_output.body_pose_6d = matrix_to_rotation_6d(body_pose_mat).reshape(
                1, T, -1
            )
            if object_pose_R_mat.numel() > 0 and object_pose_t.numel() > 0:
                sequence_output.T_world_obj = SE3.from_rotation_and_translation(
                    SO3.from_matrix(object_pose_R_mat), object_pose_t
                ).parameters()

                sequence_output.obj_Rt_6d = sequence_output.convert_object_from_global(
                    T_world_obj=sequence_output.T_world_obj,
                    T_world_head=sequence_output.T_world_head[0, :T],
                    T_can_world=sequence_output.T_can_world[0, :T],
                    target_object_pose=self.object_pose,
                ).unsqueeze(0)

            if diff_contacts.numel() > 0:
                sequence_output.diff_contacts = diff_contacts

            # Ensure the length is correct
            for k, v in sequence_output.items():
                if v is not None:
                    if k not in ["real_slice_len", "obj_class_id"]:
                        sequence_output[k] = v[:, :T]

            self._reset()
        else:
            sequence_output = EgoModelOutput()

        return sequence_output
