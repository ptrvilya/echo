import random
from dataclasses import dataclass
from logging import getLogger
from typing import Callable, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from echo.configs import ProjectConfig


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    initial_step: int = 0  # for resuming the training


def using_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
    return torch.distributed.get_rank() if using_distributed() else 0


def set_seed(seed: int, deterministic: bool = False):
    rank = get_rank()
    seed = seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    if using_distributed():
        print(f"Seeding node {rank} with seed {seed}", flush=True)
    else:
        print(f"Seeding node {rank} with seed {seed}")


class ModelEMA:
    """Exponential moving average of model weights.

    The shadow is a plain model state dict, so checkpoints store it under
    "model_ema" and it loads directly via model.load_state_dict().
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.num_updates += 1
        # warmup: effective decay ramps up from 0 over the first updates
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        # keep the shadow on the model's device (checkpoints load to cpu)
        self.shadow = {
            k: v.to(self.shadow[k].device).clone()
            for k, v in state_dict["shadow"].items()
        }


def get_optimizer(cfg: ProjectConfig, model: torch.nn.Module):
    """Gets optimizer from config"""
    logger = getLogger(__name__)
    # Determine the learning rate
    if cfg.optimizer.scale_learning_rate_with_batch_size:
        lr = cfg.dataloader.batch_size * cfg.optimizer.lr
        logger.info(
            "lr = {ws} (num gpus) * {bs} (batch_size) * {blr} (base learning rate) = {lr}".format(
                ws=0,
                bs=cfg.dataloader.batch_size,
                blr=cfg.optimizer.lr,
                lr=lr,
            )
        )
    else:  # scale base learning rate by batch size
        lr = cfg.optimizer.lr
        logger.info("lr = {lr} (absolute learning rate)".format(lr=lr))

    # Get optimizer parameters, excluding certain parameters from weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # Construct optimizer
    if cfg.optimizer.type == "torch":
        Optimizer: torch.optim.Optimizer = getattr(torch.optim, cfg.optimizer.name)
        optimizer = Optimizer(
            parameters, lr=lr, **OmegaConf.to_container(cfg.optimizer.kwargs)
        )
    else:
        raise NotImplementedError(f"Invalid optimizer config: {cfg.optimizer}")

    return optimizer


def get_scheduler(cfg: ProjectConfig, optimizer: torch.optim.Optimizer) -> Callable:
    """Gets scheduler from config"""

    # Get scheduler
    if cfg.scheduler.type == "torch":
        Scheduler: torch.optim.lr_scheduler._LRScheduler = getattr(
            torch.optim.lr_scheduler, cfg.scheduler.type
        )
        scheduler = Scheduler(
            optimizer=optimizer, **OmegaConf.to_container(cfg.scheduler.kwargs)
        )
        if cfg.scheduler.get("warmup", 0):
            from warmup_scheduler import GradualWarmupScheduler

            # pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=cfg.scheduler.warmup,
                after_scheduler=scheduler,
            )
    elif cfg.scheduler.type == "transformers":
        from transformers import get_scheduler

        scheduler = get_scheduler(
            optimizer=optimizer,
            name=cfg.scheduler.name,
            **OmegaConf.to_container(cfg.scheduler.kwargs),
        )
    else:
        raise NotImplementedError(f"invalid scheduler config: {cfg.scheduler}")

    return scheduler


def resume_from_checkpoint(
    cfg: ProjectConfig,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Callable],
) -> tuple[TrainState, Optional[dict]]:
    logger = getLogger(__name__)
    # Initialize training state
    train_state = TrainState()

    # Check if resuming training from a checkpoint
    if cfg.resume.checkpoint is None:
        logger.info("Starting training from scratch")
        return train_state, None

    # Load state dict for the model
    logger.info(f"Loading checkpoint from {cfg.resume.checkpoint}")
    checkpoint = torch.load(
        cfg.resume.checkpoint, map_location="cpu", weights_only=False
    )
    ema_state = checkpoint.get("model_ema") or None

    # At inference prefer EMA weights when available; training resumes from the
    # raw weights and restores the EMA shadow separately
    if "train" not in cfg.run.job and cfg.resume.use_ema and ema_state is not None:
        state_dict = ema_state["shadow"]
        logger.info("Using EMA weights from checkpoint")
    else:
        state_dict = checkpoint["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded model checkpoint from {cfg.resume.checkpoint}")
    if len(missing_keys):
        logger.info(f" - Missing_keys: {missing_keys}")
    if len(unexpected_keys):
        logger.info(f" - Unexpected_keys: {unexpected_keys}")

    # Load optimizer and training state (only for training)
    if "train" in cfg.run.job:
        if cfg.resume.training:
            assert (
                cfg.resume.training_optimizer
                or cfg.resume.training_scheduler
                or cfg.resume.training_state
            ), f"Invalid config: {cfg.resume}"
            # Optimizer
            if cfg.resume.training_optimizer:
                assert "optimizer" in checkpoint, f"Value not in {checkpoint.keys()}"
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Loaded optimizer from checkpoint")
            else:
                logger.info("Did not load optimizer from checkpoint")
            # Scheduler
            if cfg.resume.training_scheduler:
                assert "scheduler" in checkpoint, f"Value not in {checkpoint.keys()}"
                scheduler.load_state_dict(checkpoint["scheduler"])
                logger.info("Loaded scheduler from checkpoint")
            else:
                logger.info("Did not load scheduler from checkpoint")
            # Training state
            if cfg.resume.training_state:
                assert {"epoch", "step"}.issubset(set(checkpoint.keys()))
                epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
                train_state = TrainState(epoch=epoch, step=step, initial_step=step)
                logger.info(
                    f"Resumed state from checkpoint: step {step}, epoch {epoch}"
                )
            else:
                logger.info("Did not load train state from checkpoint")
        else:
            logger.info("Did not resume optimizer, scheduler, or epoch from checkpoint")

    logger.info("Finished loading checkpoint")

    return train_state, ema_state


def compute_grad_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def params_to_string(n_params):
    if n_params > 1e6:
        size = f"{n_params / 1e6:.1f}M"
    elif n_params > 1e3:
        size = f"{n_params / 1e3:.1f}K"
    else:
        size = f"{n_params}"
    return size


def compute_model_size(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    all_params = sum(p.numel() for p in model.parameters())

    return params_to_string(trainable_params), params_to_string(all_params)


def log_model_summary(model: torch.nn.Module, logger=None):
    if logger is None:
        logger = getLogger(__name__)

    lines = []
    lines.append("Model Summary:")
    lines.append(f"{'Layer':<80} | {'Params':<6} | {'Trainable':<6}")
    lines.append("-" * 115)

    for name, module in model.named_modules():
        # Check if module has parameters
        params = list(module.parameters(recurse=False))
        if len(params) > 0:
            num_params = sum(p.numel() for p in params)
            num_trainable = sum(p.numel() for p in params if p.requires_grad)
            lines.append(
                f"{name:<80} | {params_to_string(num_params):<6} | {params_to_string(num_trainable):<6}"
            )

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines.append("-" * 115)
    lines.append(
        f"{'Total':<80} | {params_to_string(total_params):<6} | {params_to_string(total_trainable):<6}"
    )
    logger.info("\n".join(lines))


def simulate_observation_noise(
    hmd_condition: torch.Tensor, noise_simulation_percent: int
) -> torch.Tensor:
    """Simulates observation noise by overwriting a share of the data with the previous timestep's data"""
    T = hmd_condition.shape[0]
    num_to_overwrite = min(round(T * (noise_simulation_percent / 100.0)), T - 1)

    # 1. Randomly pick indices to overwrite (excluding index 0)
    chosen_indices = np.random.choice(
        np.arange(1, T), size=num_to_overwrite, replace=False
    )

    # 2. Create a boolean mask of elements to KEEP
    keep_mask = np.ones(T, dtype=bool)
    keep_mask[chosen_indices] = False

    # 3. Create an array of positional indices: [0, 1, 2, 3, 4]
    source_indices = np.arange(T)

    # 4. Zero out the indices of elements we want to overwrite
    # Example: if keeping 0, 1, 3, 4... source_indices becomes [0, 1, 0, 3, 4]
    source_indices[~keep_mask] = 0

    # 5. Forward fill the valid indices using accumulation
    # Because indices are strictly increasing, np.maximum.accumulate pushes
    # the last valid (highest) index forward over the zeros.
    # [0, 1, 0, 3, 4] -> np.maximum.accumulate -> [0, 1, 1, 3, 4]
    fill_indices = np.maximum.accumulate(source_indices)

    # 6. Apply the calculated indices to the original array all at once
    hmd_condition = hmd_condition[fill_indices]

    return hmd_condition
