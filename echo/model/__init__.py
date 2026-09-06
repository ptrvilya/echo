from logging import getLogger

import torch
from omegaconf import OmegaConf

from echo.configs import ProjectConfig
from echo.utils.training import compute_model_size

from .model_hoic import EgoHOICModel

logger = getLogger(__name__)


def get_model(cfg: ProjectConfig):
    if cfg.model_denoising.name != "hoic_xdit":
        raise ValueError(f"Unknown model: {cfg.model_denoising.name}")
    model = EgoHOICModel(
        **OmegaConf.to_container(cfg.model, resolve=True),
        denoising_model_config=cfg.model_denoising,
        conditioning_model_config=cfg.model_conditioning,
    )
    model.init_weights()

    mixed_precision = cfg.train.mixed_precision
    if mixed_precision == "bf16":
        model.denoising_model.autocast_dtype = torch.bfloat16
        logger.info("Denoising model runs under bf16 autocast")
    elif mixed_precision != "no":
        raise NotImplementedError(
            f"mixed_precision='{mixed_precision}' is not supported, use 'no' or 'bf16'"
        )

    n_params = compute_model_size(model)
    logger.info(
        f"Created model: {cfg.model_denoising.name} with {n_params[0]} trainable params ({n_params[1]} total)."
    )

    return model
