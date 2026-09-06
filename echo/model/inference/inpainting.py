from typing import Optional, Tuple, Union

import torch
from diffusers.schedulers import SchedulerMixin

from echo.data import BatchData
from echo.model.inference.naive import InferenceModelNaive
from echo.model.output import EgoModelOutput


class InferenceInpainter(InferenceModelNaive):
    """Assumes batch size == 1"""

    def __init__(
        self,
        window_size: int,
        num_inference_steps: int = 100,
        dim_contacts: int = 0,
        overlap_size: int = 64,
        num_joints: int = 21,
        object_pose="relative",
        alpha: float = 0.0,
        prefix_conditioning: bool = False,
    ):
        super().__init__(
            window_size=window_size,
            num_inference_steps=num_inference_steps,
            object_pose=object_pose,
        )
        self.alpha = alpha
        self.prefix_conditioning = prefix_conditioning
        self.overlap_size = overlap_size
        self.window_size = window_size
        self.num_joints = num_joints
        self.object_pose = object_pose
        self.dim_contacts = dim_contacts
        self.num_inference_steps = num_inference_steps

        # accumulate in rotation matrices for averaging
        # after averaging project back to rotation matrix via svd
        self._reset()

    def _reset(self) -> None:
        # main inference data
        self.sequence_body_pose_6d = torch.empty(0)
        self.sequence_obj_pose_T_world_object = torch.empty(0)
        self.sequence_contacts = torch.empty(0)

        # noise buffer
        self.sequence_noise = torch.empty(0)

        # placeholder to accumulate output data and conditioning
        self.sequence_output: EgoModelOutput = EgoModelOutput()

        return None

    def _initialize(
        self, sequence_len: int, dim_noise: int, device: torch.device
    ) -> None:
        self.sequence_body_pose_6d = torch.empty(
            (0, self.num_joints * 6), dtype=torch.float32, device=device
        )
        self.sequence_obj_pose_T_world_object = torch.empty(
            (0, 7), dtype=torch.float32, device=device
        )
        self.sequence_contacts = torch.zeros(
            (0, self.dim_contacts), dtype=torch.float32, device=device
        )

        self.sequence_noise = torch.empty(
            (1, 0, dim_noise), dtype=torch.float32, device=device
        )

        return None

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
        if self.sequence_body_pose_6d.numel() == 0:
            total_sequence_len = batch.total_sequence_len[0].item()
            device = batch.body_pose_6d.device
            self._initialize(total_sequence_len, noise.shape[-1], device=device)

        # Reuses the noise for the overlapping region
        if self.sequence_noise.numel() == 0:
            self.sequence_noise = noise.clone()

            modified_noise = noise
        else:
            modified_noise = torch.cat(
                [
                    self.sequence_noise[:, -self.overlap_size :],
                    noise[:, self.overlap_size :],
                ],
                dim=1,
            )

            self.sequence_noise = torch.cat(
                [self.sequence_noise, noise[:, self.overlap_size :].clone()],
                dim=1,
            )

        # Provide the previous prediction as a clean prefix through the
        # per-frame conditioning (last dim_x + 1 channels: x0, known mask)
        if self.prefix_conditioning and self.sequence_body_pose_6d.numel() > 0:
            x0 = self._overlap_x0()
            K, D = x0.shape
            condition_frame = condition_frame.clone()
            condition_frame[:, :K, -D - 1 : -1] = x0[None]
            condition_frame[:, :K, -1] = 1.0

        return modified_noise, condition_frame, condition_global, t_index

    def _overlap_x0(self) -> torch.Tensor:
        """Previous prediction for the overlapping frames in network format."""
        overlapped_pred = [self.sequence_body_pose_6d[-self.overlap_size :]]
        if self.sequence_obj_pose_T_world_object.numel() > 0:
            # Convert object pose from global to network's format
            obj_Rt_6d = self.sequence_output.convert_object_from_global(
                T_world_obj=self.sequence_obj_pose_T_world_object[
                    -self.overlap_size :
                ],
                T_world_head=self.sequence_output.T_world_head[
                    0, -self.overlap_size :
                ],
                T_can_world=self.sequence_output.T_can_world[
                    0, -self.overlap_size :
                ],
                target_object_pose=self.object_pose,
            )
            overlapped_pred.append(obj_Rt_6d)
        if self.sequence_contacts.numel() > 0:
            overlapped_pred.append(self.sequence_contacts[-self.overlap_size :])
        return torch.cat(overlapped_pred, dim=-1)

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
        # Overwrite the overlapping region with the previous prediction
        if self.sequence_body_pose_6d.numel() == 0:
            modified_pred = pred
        else:
            overlapped_pred = self._overlap_x0()[None]

            if self.alpha > 1e-3:
                modified_pred = pred
                modified_pred[:, : self.overlap_size] = (
                    self.alpha * overlapped_pred
                    + (1 - self.alpha) * pred[:, : self.overlap_size]
                )
            else:
                modified_pred = torch.cat(
                    [overlapped_pred, pred[:, self.overlap_size :]], dim=1
                )

        return modified_pred

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

    def after_inference(self, batch: BatchData, x_t: torch.Tensor) -> EgoModelOutput:
        # unpack output
        output = EgoModelOutput().unpack(
            x_t,
            timesteps_motion=torch.zeros(1, x_t.shape[1]),
            batch=batch,
            object_pose=self.object_pose,
        )

        # Determine the real length of the data
        segment_length = batch.real_slice_len[0].item()

        # Select correct part of the sequence
        t_start = batch.t_start.item()
        if t_start > 0:
            offset = self.overlap_size
        else:
            offset = 0

        # Accumulated predictions
        # Assumes batch_size == 1
        body_pose_6d = output.body_pose_6d.squeeze(0)
        if offset > 0 and self.alpha > 1e-3:
            blended_pose = torch.cat(
                [
                    self.alpha * self.sequence_body_pose_6d[-self.overlap_size :]
                    + (1 - self.alpha) * body_pose_6d[: self.overlap_size],
                    body_pose_6d[self.overlap_size : segment_length],
                ],
                dim=0,
            )
        else:
            blended_pose = torch.cat(
                [
                    self.sequence_body_pose_6d[-offset:],
                    body_pose_6d[offset:segment_length],
                ],
                dim=0,
            )
        self.sequence_body_pose_6d = torch.cat(
            [self.sequence_body_pose_6d[:-offset], blended_pose],
            dim=0,
        )
        if output.T_world_obj is not None:
            object_pose = output.T_world_obj.squeeze(0)

            if offset > 0 and self.alpha > 1e-3:
                blended_pose = torch.cat(
                    [
                        self.alpha
                        * self.sequence_obj_pose_T_world_object[-self.overlap_size :]
                        + (1 - self.alpha) * object_pose[: self.overlap_size],
                        object_pose[self.overlap_size : segment_length],
                    ],
                    dim=0,
                )
            else:
                blended_pose = torch.cat(
                    [
                        self.sequence_obj_pose_T_world_object[-offset:],
                        object_pose[offset:segment_length],
                    ],
                    dim=0,
                )
            self.sequence_obj_pose_T_world_object = torch.cat(
                [self.sequence_obj_pose_T_world_object[:-offset], blended_pose],
                dim=0,
            )
        if output.diff_contacts is not None:
            contacts = output.diff_contacts.squeeze(0)
            if offset > 0 and self.alpha > 1e-3:
                blended_contacts = torch.cat(
                    [
                        self.alpha * self.sequence_contacts[-self.overlap_size :]
                        + (1 - self.alpha) * contacts[: self.overlap_size],
                        contacts[self.overlap_size : segment_length],
                    ],
                    dim=0,
                )
            else:
                blended_contacts = torch.cat(
                    [
                        self.sequence_contacts[-offset:],
                        contacts[offset:segment_length],
                    ],
                    dim=0,
                )
            self.sequence_contacts = torch.cat(
                [self.sequence_contacts[:-offset], blended_contacts],
                dim=0,
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
                            [
                                self.sequence_output[k],
                                v[:, self.overlap_size : segment_length],
                            ],
                            dim=1,
                        )
        # Determine if we return the result or skip
        if batch.is_end_of_sequence[0].item():
            body_pose_6d = self.sequence_body_pose_6d.clone()

            # Construct final output
            T = body_pose_6d.shape[0]
            sequence_output = self.sequence_output
            sequence_output.body_pose_6d = body_pose_6d.unsqueeze(0)

            if self.sequence_obj_pose_T_world_object.numel() > 0:
                T_world_obj = self.sequence_obj_pose_T_world_object
                sequence_output.T_world_obj = T_world_obj.unsqueeze(0)
                sequence_output.obj_Rt_6d = sequence_output.convert_object_from_global(
                    T_world_obj=T_world_obj,
                    T_world_head=sequence_output.T_world_head[0, :T],
                    T_can_world=sequence_output.T_can_world[0, :T],
                    target_object_pose=self.object_pose,
                ).unsqueeze(0)

            if self.sequence_contacts.numel() > 0:
                diff_contacts = self.sequence_contacts
                sequence_output.diff_contacts = diff_contacts.unsqueeze(0)

            # Ensure the length is correct
            for k, v in sequence_output.items():
                if v is not None:
                    if k not in ["real_slice_len", "obj_class_id"]:
                        sequence_output[k] = v[:, :T]

            self._reset()
        else:
            sequence_output = EgoModelOutput()

        return sequence_output
