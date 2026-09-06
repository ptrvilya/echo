import math
import sys
import time
from collections import defaultdict
from logging import getLogger
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from echo.configs import ProjectConfig
from echo.data import get_train_dataloaders
from echo.data.batch_data import BatchData
from echo.data.normalization import denormalize_obj_Rt_6d
from echo.model.base import BaseEgoModel
from echo.model.inference import InferenceModelNaive
from echo.model.mesh_model import MeshModel
from echo.model.output import EgoModelOutput
from echo.utils.metrics.reconstruction import get_mpjpe, get_mpjve
from echo.utils.training import (
    ModelEMA,
    compute_grad_norm,
    get_optimizer,
    get_scheduler,
    resume_from_checkpoint,
)
from echo.utils.transforms import SE3, SO3


class Trainer:
    def __init__(self, config: ProjectConfig, model: BaseEgoModel):
        super().__init__()

        self.config = config
        self.model = model.cuda()
        self.device = torch.device("cuda")

        if config.train.compile_denoiser:
            # compile the bound method, not the module: wrapping the module
            # prefixes state-dict keys with `_orig_mod` and breaks the EMA
            denoiser = self.model.denoising_model.model
            denoiser.forward = torch.compile(denoiser.forward)

        optimizer = get_optimizer(config, model)
        scheduler = get_scheduler(config, optimizer)

        # Resume from checkpoint and create the initial training state
        self.train_state, ema_state = resume_from_checkpoint(
            config, model, optimizer, scheduler
        )

        # EMA of model weights (created after resume so the shadow starts from
        # the loaded weights; restored from the checkpoint when available)
        self.ema: ModelEMA | None = None
        if config.train.ema_decay:
            self.ema = ModelEMA(model, decay=config.train.ema_decay)
            if ema_state is not None:
                self.ema.load_state_dict(ema_state)

        # Get dataloaders
        dataloader_train, dataloader_val, obj_keypoints = get_train_dataloaders(config)

        # Create mesh model to obtain posed joints
        self.mesh_model = MeshModel(
            model_type=config.model_mesh.type,
            env=config.env,
            num_betas=config.model_mesh.num_betas,
            include_object=config.model_mesh.include_object,
            device=self.device,
            inference_chunk_size=config.model_mesh.inference_chunk_size,
            batch_size=config.model_mesh.inference_chunk_size,
            canonical_obj_keypoints=obj_keypoints,
            infer_joints_only=not self.config.model_mesh.with_skin,
        )

        # Validation always runs naive per-window inference on full batches;
        # the stitching/inpainting models assume batch_size == 1
        self.inference_model = InferenceModelNaive(
            window_size=config.model.window_size,
            num_inference_steps=config.sample.num_inference_steps,
            object_pose=config.model.object_pose,
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.logger = getLogger(__name__)

    def get_outputs(
        self,
        batch: BatchData,
        mode: str = "train",
        masked_inference=False,
        infer_meshes=True,
        **kwargs,
    ) -> EgoModelOutput:
        assert mode in ["train", "sample_naive"], f"Invalid mode: {mode}"
        # aux_output is (x_0, x_t, noise, x_0_pred)
        if mode == "sample_naive":
            kwargs.update(
                {
                    "sample_type": self.config.sample.modalities,
                    "inference_model": self.inference_model,
                    "num_inference_steps": self.config.sample.num_inference_steps,
                }
            )

        output: EgoModelOutput = self.model(batch, mode, **kwargs)

        # get predicted meshes
        if infer_meshes:
            # mask for hoi-based losses
            with torch.no_grad():
                B, T = batch.body_pose_6d.shape[:2]
                mask_t = (
                    output.timesteps_motion <= self.config.train.loss_t_stamp_threshold
                )
                mask_t = mask_t.view(-1, 1).expand(B, T).reshape(-1)

                mesh_output = self.mesh_model.get_meshes_from_batch(
                    batch,
                    with_skin=self.config.model_mesh.with_skin,
                    inference_mask=mask_t if masked_inference else None,
                )
                batch.sbj_joints = mesh_output.sbj_joints
                batch.obj_keypoints = mesh_output.obj_keypoints

                if self.config.model_mesh.with_skin:
                    batch.sbj_vertices = mesh_output.sbj_vertices[
                        :, :, self.mesh_model.smpl_downsample_indices
                    ]

            # join mask with t_stamp based mask
            mesh_output = self.mesh_model.get_meshes_from_output(
                output,
                return_joints=True,
                local_to_global=True,
                with_skin=self.config.model_mesh.with_skin,
                inference_mask=mask_t if masked_inference else None,
            )
            output.sbj_joints = mesh_output.sbj_joints
            output.obj_keypoints = mesh_output.obj_keypoints

            if self.config.model_mesh.with_skin:
                output.sbj_vertices = mesh_output.sbj_vertices[
                    :, :, self.mesh_model.smpl_downsample_indices
                ]

        return output

    def compute_loss(
        self,
        batch: BatchData,
        output: EgoModelOutput,
        step: int = -1,
        log_details: bool = True,
    ):
        wandb_log = dict()
        loss = 0
        # Threshold for auxiliary losses on prediction
        THR = self.config.train.loss_t_stamp_threshold

        # Linear ramp-in of aux losses after warmup, to avoid hitting the
        # converged model with a step function of new gradients.
        # Validation (step=-1) stays unscaled so VAL curves are comparable.
        AUX_KEYS = ("obj_smooth", "obj_velocity", "sbj_joints", "sbj_skate")
        aux_ramp = 1.0
        ramp_steps = self.config.train.aux_loss_ramp_steps
        if step > -1 and ramp_steps > 0:
            warmup = self.config.train.aux_loss_warmup_steps
            aux_ramp = min(1.0, max(0.0, (step - warmup) / ramp_steps))

        mask_hoi = (~batch.is_fake_object.type(torch.bool)).to(
            output.body_pose_6d.device
        )

        for key, weight in self.config.train.losses.items():
            # denoising loss
            if key == "denoise_motion":
                loss_i = F.mse_loss(
                    output.body_pose_6d,
                    batch.body_pose_6d,
                    reduction="none",
                ).mean(dim=(1, 2))
            elif key == "denoise_object":
                if not mask_hoi.any():
                    loss_i = torch.tensor(0.0).to(output.body_pose_6d.device)
                else:
                    # average over real-object samples only, so the effective
                    # weight does not depend on the batch composition
                    pred = output.obj_Rt_6d[mask_hoi]
                    gt = batch.obj_Rt_6d[mask_hoi]

                    loss_i = F.mse_loss(pred, gt, reduction="none").mean(dim=(1, 2))
            elif key == "denoise_contact_env_mse":
                pred_contact = output.diff_contacts[
                    :, :, self.config.model.dim_contacts_hoi :
                ]
                gt_contacts = batch.contacts[:, :, self.config.model.dim_contacts_hoi :]

                loss_i = F.mse_loss(
                    pred_contact,
                    gt_contacts,
                    reduction="none",
                ).mean(dim=(1, 2))
            elif key == "denoise_contact_hoi_mse":
                if not mask_hoi.any():
                    loss_i = torch.tensor(0.0).to(output.body_pose_6d.device)
                else:
                    pred_contact = output.diff_contacts[
                        mask_hoi, :, : self.config.model.dim_contacts_hoi
                    ]
                    gt_contacts = batch.contacts[
                        mask_hoi, :, : self.config.model.dim_contacts_hoi
                    ]
                    loss_i = F.mse_loss(
                        pred_contact,
                        gt_contacts,
                        reduction="none",
                    ).mean(dim=(1, 2))
            elif key == "sbj_joints":
                if step > -1 and step < self.config.train.aux_loss_warmup_steps:
                    loss_i = torch.tensor(0.0).to(output.body_pose_6d.device)
                else:
                    # only apply loss for cleaner samples
                    mask_t = output.timesteps_motion <= THR
                    mask_t = mask_t.to(output.body_pose_6d.device)
                    B, T = output.body_pose_6d.shape[:2]
                    mask_t = mask_t.view(B)

                    end_effectors_ids = torch.tensor(
                        [7, 8, 10, 11, 18, 19, 20, 21],
                        dtype=torch.long,
                        device=self.device,
                    )

                    pred_ee_joints = output.sbj_joints[:, :, end_effectors_ids]
                    gt_ee_joints = batch.sbj_joints[:, :, end_effectors_ids]

                    # L1 loss on joint positions
                    if mask_t.any():
                        loss_pos = F.l1_loss(
                            pred_ee_joints[mask_t],
                            gt_ee_joints[mask_t],
                            reduction="none",
                        ).mean(dim=(1, 2, 3))
                    else:
                        loss_pos = torch.tensor(0.0).to(output.body_pose_6d.device)

                    # L1 loss on joint velocities to enforce temporal smoothness
                    pred_vel = pred_ee_joints[:, 1:] - pred_ee_joints[:, :-1]
                    gt_vel = gt_ee_joints[:, 1:] - gt_ee_joints[:, :-1]

                    if mask_t.any():
                        loss_vel = F.l1_loss(
                            pred_vel[mask_t], gt_vel[mask_t], reduction="none"
                        ).mean(dim=(1, 2, 3))
                    else:
                        loss_vel = torch.tensor(0.0).to(output.body_pose_6d.device)

                    # velocity term carries ~2x the gradient of the position
                    # term; keep it a small fraction
                    loss_i = loss_pos + 0.1 * loss_vel
            elif key == "obj_smooth":
                if step > -1 and step < self.config.train.aux_loss_warmup_steps:
                    loss_i = torch.tensor(0.0).to(output.body_pose_6d.device)
                else:
                    # penalty for jitter applies only on later stages of object diffusion
                    mask_t = output.timesteps_object <= THR
                    mask_t = mask_t.to(output.obj_Rt_6d.device)
                    # mask_hoi is B, mask_t is BxT
                    B, T = output.body_pose_6d.shape[:2]
                    mask_t = mask_t.view(B, -1)
                    mask = torch.logical_and(mask_hoi[:, None], mask_t).float()
                    mask = mask.expand(B, T)

                    # B x T x 9
                    T_world_obj = SE3(output.T_world_obj).as_matrix()
                    obj_R, obj_t = T_world_obj[:, :, :3, :3], T_world_obj[:, :, :3, 3]

                    # translation acceleration
                    velocity = obj_t[:, 1:] - obj_t[:, :-1]
                    acceleration = velocity[:, 1:] - velocity[:, :-1]

                    # angular acceleration as the difference of angular
                    # velocities w_t = log(R_t^T @ R_t+1) in the tangent space
                    angular_velocity = SO3.from_matrix(
                        torch.matmul(obj_R[:, :-1].transpose(2, 3), obj_R[:, 1:])
                    ).log()
                    angular_acceleration = (
                        angular_velocity[:, 1:] - angular_velocity[:, :-1]
                    )

                    mask_acc = mask[:, 1:-1]
                    denom = mask_acc.sum().clamp(min=1.0)
                    loss_i = (
                        2 * (torch.norm(acceleration, dim=2) * mask_acc).sum()
                        + (torch.norm(angular_acceleration, dim=2) * mask_acc).sum()
                    ) / denom
            elif key == "obj_velocity":
                if step > -1 and step < self.config.train.aux_loss_warmup_steps:
                    loss_i = torch.tensor(0.0).to(output.body_pose_6d.device)
                else:
                    # prior: an object cannot move while it is not in contact
                    mask_t = output.timesteps_object <= THR
                    mask_t = mask_t.to(output.obj_Rt_6d.device)
                    B, T = output.body_pose_6d.shape[:2]
                    mask_t = mask_t.view(B, -1)
                    mask = torch.logical_and(mask_hoi[:, None], mask_t)
                    T = output.obj_Rt_6d.shape[1]
                    mask = mask.expand(B, T)

                    # require no GT contact at both ends of the velocity
                    # interval (same two-frame convention as sbj_skate)
                    gt_contact = (
                        batch.contacts[:, :, : self.config.model.dim_contacts_hoi].amax(
                            dim=2
                        )
                        > 0.5
                    )
                    no_contact = torch.logical_and(
                        ~gt_contact[:, 1:], ~gt_contact[:, :-1]
                    )
                    mask = (mask[:, 1:] & no_contact).float()

                    # B x T x 9
                    T_world_obj = SE3(output.T_world_obj).as_matrix()
                    obj_R, obj_t = T_world_obj[:, :, :3, :3], T_world_obj[:, :, :3, 3]

                    # any motion of an untouched object is penalized
                    velocity_magnitude = torch.norm(
                        obj_t[:, 1:] - obj_t[:, :-1], dim=-1
                    )  # B x T-1
                    rotation_angle = torch.norm(
                        SO3.from_matrix(
                            torch.matmul(obj_R[:, :-1].transpose(2, 3), obj_R[:, 1:])
                        ).log(),
                        dim=-1,
                    )  # B x T-1

                    denom = mask.sum().clamp(min=1.0)
                    loss_i = (
                        5 * (velocity_magnitude * mask).sum()
                        + (rotation_angle * mask).sum()
                    ) / denom
            elif key == "sbj_skate":
                if step > -1 and step < self.config.train.aux_loss_warmup_steps:
                    loss_i = torch.tensor(0.0).to(output.body_pose_6d.device)
                else:
                    assert self.config.model.include_env_contact

                    # only apply loss for cleaner samples
                    mask_t = output.timesteps_motion <= THR
                    mask_t = mask_t.to(output.body_pose_6d.device)
                    B, T = output.body_pose_6d.shape[:2]
                    mask_t = mask_t.view(B, -1).expand(B, T)

                    # Extract feet contacts from environment contacts
                    # env_contacts are at the end of batch.contacts, following ENV_CONTACT_JOINTS order.
                    # Feet joints (ankle, foot) are at indices 2, 3, 4, 5 in ENV_CONTACT_JOINTS.
                    dim_hoi = self.config.model.dim_contacts_hoi
                    gt_contact_label = batch.contacts[
                        :, :, dim_hoi + 2 : dim_hoi + 6
                    ].to(output.body_pose_6d.device)

                    # Use contact labels from both frames to ensure the foot is stationary during the transition
                    mask_contact = (
                        gt_contact_label[:, 1:] * gt_contact_label[:, :-1]
                    ) > 0.5
                    mask_t = mask_t[:, :-1].unsqueeze(2) & mask_contact

                    pred_joints = output.sbj_joints
                    feet_indices = torch.tensor(
                        [7, 8, 10, 11], dtype=torch.long, device=pred_joints.device
                    )
                    feet_joints = pred_joints[:, :, feet_indices]
                    feet_velocity = feet_joints[:, 1:] - feet_joints[:, :-1]

                    if mask_t.any():
                        # squared speed: gradient decays as sliding -> 0,
                        # unlike torch.norm whose gradient is a unit vector
                        loss_i = feet_velocity[mask_t].pow(2).sum(-1).mean()
                    else:
                        loss_i = torch.tensor(0.0, device=output.body_pose_6d.device)
            else:
                raise NotImplementedError(f"No implementation for {key} loss.")

            if isinstance(loss_i, torch.Tensor):
                loss_i = loss_i.mean()
            loss += weight * (aux_ramp if key in AUX_KEYS else 1.0) * loss_i
            # per-key .item() forces a host sync; skip on non-logging steps
            if log_details:
                wandb_log[f"loss/{key}"] = loss_i.detach().cpu().item()

        wandb_log["loss/total"] = loss.detach().cpu().item()
        return loss, wandb_log

    def save_checkpoint(self):
        checkpoint_dict = {
            "model": (self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.train_state.epoch,
            "step": self.train_state.step,
            "relative_step": self.train_state.step - self.train_state.initial_step,
            "model_ema": self.ema.state_dict() if self.ema else {},
            "cfg": OmegaConf.to_container(self.config, resolve=True),
        }

        checkpoint_name = f"checkpoint-step-{self.train_state.step:07d}.pth"
        checkpoint_path = Path(self.config.run.path) / "checkpoints" / checkpoint_name
        torch.save(checkpoint_dict, checkpoint_path)
        self.logger.info(f"\nSaved checkpoint to {checkpoint_path}")

    def train(self):
        total_batch_size = self.config.dataloader.batch_size

        # Log info
        self.logger.info(
            f"***** Starting training *****\n"
            f"    Dataset train size: {len(self.dataloader_train.dataset):_}\n"
            f"    Dataset val size: {len(self.dataloader_val.dataset):_}\n"
            f"    Dataloader train size: {len(self.dataloader_train):_}\n"
            f"    Dataloader val size: {len(self.dataloader_val):_}\n"
            f"    Batch size per device = {self.config.dataloader.batch_size}\n"
            f"    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n"
            f"    Gradient Accumulation steps = {self.config.optimizer.gradient_accumulation_steps}\n"
            f"    Max training steps = {self.config.train.max_steps}\n"
            f"    Training state = {self.train_state}"
        )

        while True:
            # Train loop
            self.model.train()
            for i, batch in tqdm(
                enumerate(self.dataloader_train),
                ncols=80,
                leave=False,
                total=len(self.dataloader_train),
                desc=f"Train Ep:{self.train_state.epoch}",
            ):
                # Early stop
                if (self.config.train.limit_train_batches is not None) and (
                    i >= self.config.train.limit_train_batches
                ):
                    break

                # Process one training step
                wandb_log = self.train_step(batch, step=self.train_state.step)

                # Check if loss was None
                if not math.isfinite(wandb_log["loss/total"]):
                    self.logger.error(
                        "Loss is {}, stopping training".format(wandb_log["loss/total"])
                    )
                    self.logger.error(wandb_log)
                    sys.exit(1)

                # Log to wandb
                if (
                    self.config.logging.wandb
                    and self.train_state.step % self.config.train.log_step_freq == 0
                ):
                    wandb_log["lr"] = self.optimizer.param_groups[0]["lr"]
                    wandb_log["step"] = self.train_state.step
                    wandb_log["relative_step"] = (
                        self.train_state.step - self.train_state.initial_step
                    )
                    wandb.log(wandb_log, step=self.train_state.step)

                # Save checkpoint
                if self.train_state.step % self.config.train.checkpoint_freq == 0:
                    self.save_checkpoint()

                # Exit training
                if self.train_state.step >= self.config.train.max_steps:
                    self.logger.info(
                        f"Ending training at with state: {self.train_state}"
                    )

                    wandb.finish()
                    time.sleep(5)
                    return

            # Val loop
            if self.train_state.epoch % self.config.train.val_freq == 0:
                val_log = self.run_validation(prefix="VAL_")

                # validate EMA weights as well (temporarily swapped into the model)
                if self.ema is not None:
                    backup = {
                        k: v.detach().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    self.model.load_state_dict(self.ema.shadow)
                    val_log.update(self.run_validation(prefix="VAL_EMA_"))
                    self.model.load_state_dict(backup)

                val_log["epoch"] = self.train_state.epoch
                if self.config.logging.wandb:
                    wandb.log(val_log, step=self.train_state.step)

            # Train epoch complete
            self.train_state.epoch += 1

    def run_validation(self, prefix: str = "VAL_"):
        self.model.eval()
        val_log, val_metrics, val_counters = (
            defaultdict(float),
            defaultdict(float),
            defaultdict(int),
        )
        limit = self.config.train.limit_val_batches
        num_batches = len(self.dataloader_val)
        if limit is not None:
            num_batches = min(num_batches, limit)
        for i, batch in tqdm(
            enumerate(self.dataloader_val),
            ncols=80,
            leave=False,
            total=num_batches,
            desc=f"Val Ep:{self.train_state.epoch}",
        ):
            if i >= num_batches:
                break
            val_losses, tmp_metrics, tmp_counters = self.validation_step(batch)

            for k, v in val_losses.items():
                val_log[k] += v
            for k in tmp_metrics.keys():
                val_metrics[k] += tmp_metrics[k]
                val_counters[k] += tmp_counters[k]

        val_log = {f"{prefix}{k}": v / num_batches for k, v in val_log.items()}
        for k in val_metrics.keys():
            val_log[f"{prefix}metrics/{k}"] = val_metrics[k] / (val_counters[k] + 1e-4)

        return val_log

    def train_step(self, batch: BatchData, step: int):
        self.model.train()
        batch = batch.to(self.device)
        # mesh-dependent losses are zeroed during warmup, skip the SMPL-X passes
        infer_meshes = (
            self.config.train.infer_meshes
            and step >= self.config.train.aux_loss_warmup_steps
        )
        output = self.get_outputs(
            batch,
            mode="train",
            infer_meshes=infer_meshes,
            masked_inference=True,
        )
        # +1: the outer wandb.log gate checks train_state.step after the
        # increment at the end of this function
        log_step = (step + 1) % self.config.train.log_step_freq == 0
        loss, wandb_log = self.compute_loss(batch, output, step, log_details=log_step)

        # backward
        loss.backward()
        if self.config.optimizer.clip_grad_norm is not None:
            # grad-norm computation syncs per-parameter; only do it when logging
            if log_step:
                wandb_log["grad_norm"] = compute_grad_norm(self.model.parameters())
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.optimizer.clip_grad_norm
            )
            if log_step:
                wandb_log["grad_norm_clipped"] = compute_grad_norm(
                    self.model.parameters()
                )

        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.ema is not None:
            self.ema.update(self.model)
        self.scheduler.step()
        self.train_state.step += 1

        return wandb_log

    @torch.no_grad()
    def compute_val_metrics(self, batch: BatchData, output: EgoModelOutput):
        METRICS = ["MPJPE", "MPJVE", "OBJ_CENTER"]
        tmp_metrics = {metric: 0.0 for metric in METRICS}
        tmp_counters = {metric: 0 for metric in METRICS}

        pred_joints = output.sbj_joints.detach().cpu().numpy()
        gt_joints = batch.sbj_joints.detach().cpu().numpy()

        mpjpe = get_mpjpe(pred_joints, gt_joints)
        tmp_metrics["MPJPE"] += mpjpe
        tmp_counters["MPJPE"] += batch.batch_size()
        mpjpe_pa = get_mpjve(pred_joints, gt_joints)
        tmp_metrics["MPJVE"] += mpjpe_pa
        tmp_counters["MPJVE"] += batch.batch_size()

        mask_hoi = ~batch.is_fake_object
        if output.obj_Rt_6d is not None and mask_hoi.any():
            # denormalize to report the metric in meters
            pred_obj_t = denormalize_obj_Rt_6d(output.obj_Rt_6d[mask_hoi])[:, :, 6:]
            gt_obj_t = denormalize_obj_Rt_6d(batch.obj_Rt_6d[mask_hoi])[:, :, 6:]
            obj_center_dist = (
                torch.linalg.norm(
                    pred_obj_t - gt_obj_t,
                    dim=2,
                )
                .mean(1)
                .sum(0)
                .detach()
                .cpu()
                .numpy()
            )
            tmp_metrics["OBJ_CENTER"] += obj_center_dist
            tmp_counters["OBJ_CENTER"] += mask_hoi.sum().item()

        return tmp_metrics, tmp_counters

    @torch.no_grad()
    def validation_step(self, batch: BatchData):
        # Naive inference without conveyor
        batch = batch.to(self.device)
        output: EgoModelOutput = self.get_outputs(
            batch,
            mode="sample_naive",
            scheduler=self.config.sample.scheduler,
            infer_meshes=True,
            num_inference_steps=self.config.sample.num_inference_steps,
        )
        _, wandb_log = self.compute_loss(batch, output)

        val_metrics, val_counters = self.compute_val_metrics(batch, output)

        return wandb_log, val_metrics, val_counters
