from logging import getLogger

import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from omegaconf import OmegaConf
from torch import nn

from echo.configs import ConditioningModelConfig, DenoisingModelConfig
from echo.data.batch_data import BatchData

from .conditioning.model import ConditioningModel
from .denoising.model import DenoisingModel
from .schedulers.ddpm_batched import DDPMSchedulerBatched

logger = getLogger(__name__)


def get_custom_betas(
    beta_start: float,
    beta_end: float,
    warmup_frac: float = 0.3,
    num_train_timesteps: int = 1000,
):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


class BaseEgoModel(nn.Module):
    def __init__(
        self,
        # Input configuration
        dim_motion_feature: int,
        dim_output_motion: int,
        num_joints: int,
        window_size: int,
        # diffusion parameters
        denoise_mode: str,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        # sub-models configs
        denoising_model_config: DenoisingModelConfig,
        conditioning_model_config: ConditioningModelConfig,
        # optional parameters
        dim_input_object: int | None = None,
        dim_output_object: int | None = None,
        dim_contacts: int | None = None,
        dim_contacts_hoi: int | None = None,
        dim_contacts_env: int | None = None,
        object_pose: str | None = "canonical",
        include_env_contact: bool | None = False,
        include_hoi_contact: bool | None = False,
        # classifier guidance
        cg_apply: bool = False,
        cg_scale: float = 0.0,
        cg_t_stamp: int = 200,
        cg_skate_relative_scale: float = 0.2,
        cg_contact_threshold: float = 0.05,
    ):
        super().__init__()

        self.denoise_mode = denoise_mode
        self.cg_apply = cg_apply
        self.cg_scale = cg_scale
        self.cg_t_stamp = cg_t_stamp
        self.cg_skate_relative_scale = cg_skate_relative_scale
        self.cg_contact_threshold = cg_contact_threshold
        self.object_pose = object_pose
        self.include_env_contact = include_env_contact
        self.include_hoi_contact = include_hoi_contact

        # Input size
        self.dim_motion_feature = dim_motion_feature
        self.dim_output_motion = dim_output_motion
        self.dim_input_object = dim_input_object
        self.dim_output_object = dim_output_object
        self.dim_contacts = dim_contacts
        self.dim_contacts_hoi = dim_contacts_hoi
        self.dim_contacts_env = dim_contacts_env
        self.num_joints = num_joints
        self.window_size = window_size
        self.data_in_channels = self.dim_motion_feature * self.num_joints
        self.dim_embedding = denoising_model_config.dim_embedding

        # Output size
        self.data_motion_channels = self.dim_output_motion * self.num_joints

        # Create conditioning model
        self.conditioning_model = ConditioningModel(
            window_size=self.window_size,
            dim_prefix_motion=self.data_in_channels,
            dim_prefix_object=self.dim_output_object or 0,
            dim_prefix_contacts_hoi=self.dim_contacts_hoi or 0,
            dim_prefix_contacts_env=self.dim_contacts_env or 0,
            **OmegaConf.to_container(conditioning_model_config, resolve=True),
        )
        self.cond_frame_channels = self.conditioning_model.cond_frame_channels
        self.cond_global_channels = self.conditioning_model.cond_global_channels

        # Create denoising model for processing parameters at each diffusion step
        self.denoising_model = DenoisingModel(
            name=denoising_model_config.name,
            dim_motion_feature=self.dim_motion_feature,
            dim_intermediate_embedding=denoising_model_config.dim_intermediate_embedding,
            dim_output_motion=self.dim_output_motion,
            dim_condition_frame=self.cond_frame_channels,
            dim_condition_global=self.cond_global_channels,
            num_joints=self.num_joints,
            window_size=self.window_size,
            dim_embedding=self.dim_embedding,
            dim_input_object=self.dim_input_object,
            dim_output_object=self.dim_output_object,
            dim_contacts=dim_contacts,
            dim_contacts_env=dim_contacts_env,
            dim_contacts_hoi=dim_contacts_hoi,
            **denoising_model_config.params,
        )

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == "custom":
            scheduler_kwargs.update(
                dict(
                    trained_betas=get_custom_betas(
                        beta_start=beta_start, beta_end=beta_end
                    )
                )
            )
        else:
            scheduler_kwargs.update(
                dict(
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_schedule=beta_schedule,
                )
            )
        scheduler_kwargs.update(dict(prediction_type=self.denoise_mode))
        self.schedulers_map = {
            "ddpm": DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            "ddim": DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            "pndm": PNDMScheduler(**scheduler_kwargs),
            "ddpm_b": DDPMSchedulerBatched(**scheduler_kwargs, clip_sample=False),
        }
        self.scheduler = self.schedulers_map[
            "ddpm"
        ]  # this can be changed for inference

    def init_weights(self):
        self.denoising_model.init_weights()
        self.conditioning_model.init_weights()

    def get_conditioning(self, batch: BatchData, **kwargs):
        return self.conditioning_model.get_conditioning(batch, **kwargs)

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError()

    def forward_sample(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        """A wrapper around the forward method for training and inference"""
        raise NotImplementedError()

    def set_mesh_model(self, mesh_model):
        self.mesh_model = mesh_model
