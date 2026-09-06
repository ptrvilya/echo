from typing import Optional, Tuple, Union

import torch
from diffusers.schedulers import SchedulerMixin

from echo.data import BatchData
from echo.model.output import EgoModelOutput


class InferenceModelNaive:
    def __init__(
        self,
        window_size,
        num_inference_steps: int,
        object_pose: str = "relative",
        **kwargs,
    ):
        self.window_size = window_size
        self.num_inference_steps = num_inference_steps
        self.object_pose = object_pose
        return None

    def _reset(self) -> None:
        return None

    def _initialize(self, *args, **kwargs) -> None:
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
        return noise, condition_frame, condition_global, t_index

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

    def after_inference(self, batch: BatchData, x_t: torch.Tensor) -> EgoModelOutput:
        output = EgoModelOutput().unpack(
            x_t,
            timesteps_motion=torch.zeros(x_t.shape[0]),
            batch=batch,
            object_pose=self.object_pose,
        )

        return output
