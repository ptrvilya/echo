import inspect
from copy import deepcopy
from logging import getLogger

import torch
from torch import Tensor
from tqdm import tqdm

from echo.data.batch_data import BatchData
from echo.model.base import BaseEgoModel
from echo.model.inference import InferenceModelNaive
from echo.model.output import EgoModelOutput

logger = getLogger(__name__)


class EgoHOICModel(BaseEgoModel):
    def __init__(
        self,
        **kwargs,  # conditioning arguments
    ):
        super().__init__(**kwargs)

        self.scheduler_aux_1 = deepcopy(
            self.schedulers_map["ddpm"]
        )  # auxiliary scheduler for object
        self.scheduler_aux_2 = deepcopy(
            self.schedulers_map["ddpm"]
        )  # auxiliary scheduler for contacts

        trange = torch.arange(
            0, self.scheduler.config.num_train_timesteps, dtype=torch.long
        )
        tzeros = torch.zeros_like(trange)
        # (t_motion, t_object, t_contact) blocks with sampling weights;
        # repeats out of 40: the joint denoising block (inference regime) is
        # sampled 50% of the time, object denoising with clean motion 10% each,
        # the remaining combinations 7.5% each
        blocks = [
            (torch.stack([trange, trange, trange], dim=1), 20),
            (torch.stack([tzeros, trange, tzeros], dim=1), 4),
            (torch.stack([tzeros, trange, trange], dim=1), 4),
            (torch.stack([tzeros, tzeros, trange], dim=1), 3),
            (torch.stack([trange, tzeros, tzeros], dim=1), 3),
            (torch.stack([trange, tzeros, trange], dim=1), 3),
            (torch.stack([trange, trange, tzeros], dim=1), 3),
        ]
        self.sparse_timesteps = torch.cat(
            [torch.zeros(3, dtype=torch.long).unsqueeze(0)]
            + [block for block, repeats in blocks for _ in range(repeats)],
            dim=0,
        )

    def forward_train(self, batch: BatchData, return_intermediate_steps: bool = False):
        # Parse input
        motion = batch.body_pose_6d
        _object = batch.obj_Rt_6d
        _contact = batch.contacts

        # Parse condition (optionally with a clean prefix of x_0)
        prefix_x0, prefix_mask = self.conditioning_model.sample_prefix(
            torch.cat([motion, _object, _contact], dim=2)
        )
        condition_frame, condition_global = self.get_conditioning(
            batch, prefix_x0=prefix_x0, prefix_mask=prefix_mask
        )

        # Get dimensions
        B, T, D_motion = motion.shape
        B, T, D_object = _object.shape
        B, T, D_contact = _contact.shape

        # Sample random noise
        noise_motion = torch.randn_like(motion)
        noise_object = torch.randn_like(_object)
        noise_contact = torch.randn_like(_contact)

        # Save for auxillary output
        noise = torch.cat([noise_motion, noise_object, noise_contact], dim=2)
        x_0 = torch.cat([motion, _object, _contact], dim=2)

        # sparse sampling
        timestep_indices = torch.randint(
            0, len(self.sparse_timesteps), (B,), dtype=torch.long
        )
        timesteps = self.sparse_timesteps[timestep_indices]
        timesteps_motion, timesteps_object, timesteps_contact = (
            timesteps[:, [0]],
            timesteps[:, [1]],
            timesteps[:, [2]],
        )

        # Add noise to the input
        motion_t = self.scheduler.add_noise(
            motion.reshape(B * T, D_motion),
            noise_motion.reshape(B * T, D_motion),
            timesteps_motion.expand(B, T).reshape(B * T),
        ).reshape(B, T, D_motion)
        object_t = self.scheduler_aux_1.add_noise(
            _object.reshape(B * T, D_object),
            noise_object.reshape(B * T, D_object),
            timesteps_object.expand(B, T).reshape(B * T),
        ).reshape(B, T, D_object)
        contact_t = self.scheduler_aux_2.add_noise(
            _contact.reshape(B * T, D_contact),
            noise_contact.reshape(B * T, D_contact),
            timesteps_contact.expand(B, T).reshape(B * T),
        ).reshape(B, T, D_contact)

        # Merge sbj, obj, and contact
        x_t_input = torch.cat([motion_t, object_t, contact_t], dim=2)

        # Forward
        if self.denoise_mode == "epsilon":
            raise NotImplementedError(f"Unknown denoise_mode: {self.denoise_mode}")
        elif self.denoise_mode == "sample":
            x_0_pred = self.denoising_model(
                x_t_input,
                t=timesteps_motion,
                t_object=timesteps_object,
                t_contact=timesteps_contact,
                condition_frame=condition_frame,
                condition_global=condition_global,
                mask_hoi=~batch.is_fake_object,
            )

            # Check
            assert x_0_pred.shape == x_0.shape, (
                f"Input prediction {x_0_pred.shape=} and {x_0.shape=}"
            )

            # Unpack
            prediction = EgoModelOutput().unpack(
                x_0_pred,
                timesteps_motion=timesteps_motion,
                timesteps_object=timesteps_object,
                timesteps_contact=timesteps_contact,
                batch=batch,
                object_pose=self.object_pose,
            )

            # Auxiliary output
            aux_output = (x_0, x_t_input, noise)
        else:
            raise NotImplementedError(f"Unknown denoise_mode: {self.denoise_mode}")

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return prediction, aux_output
        else:
            return prediction

    def _guidance_losses(self, mode, x0, batch):
        """Contact and foot-skate losses"""
        device = x0.device
        B, T = x0.shape[:2]
        D_motion = self.data_motion_channels
        D_object = self.dim_output_object

        # channels that are not sampled come from the batch (conditioning)
        motion = (
            x0[:, :, :D_motion] if mode[0] == "1" else batch.body_pose_6d.to(device)
        )
        obj = (
            x0[:, :, D_motion : D_motion + D_object]
            if mode[1] == "1"
            else batch.obj_Rt_6d.to(device)
        )
        contact = (
            x0[:, :, D_motion + D_object :]
            if mode[2] == "1"
            else batch.contacts.to(device)
        )
        output = EgoModelOutput().unpack(
            torch.cat([motion, obj, contact], dim=2),
            batch=batch,
            object_pose=self.object_pose,
        )

        mesh_output = self.mesh_model.get_meshes_from_output(
            output, return_joints=True, local_to_global=True, with_skin=True
        )
        sbj_vertices = mesh_output.sbj_vertices[
            :, :, self.mesh_model.smpl_downsample_indices
        ]
        sbj_joints = mesh_output.sbj_joints.reshape(B, T, -1, 3)

        # contact: pull vertices with predicted HOI contactness towards the object
        # until they are within the contact threshold
        pred_dist = (
            torch.cdist(
                sbj_vertices.reshape(B * T, -1, 3),
                mesh_output.obj_keypoints.reshape(B * T, -1, 3),
            )
            .min(dim=-1)
            .values
        )
        contactness_hoi = output.diff_contacts.reshape(B * T, -1)[
            :, : self.dim_contacts_hoi
        ].detach()
        if batch.is_fake_object is not None:
            real = (~batch.is_fake_object.to(device)).float().view(B, 1).expand(B, T)
            contactness_hoi = contactness_hoi * real.reshape(B * T, 1)
        contacts_loss = (
            contactness_hoi * (pred_dist - self.cg_contact_threshold).clamp(min=0.0)
        ).sum() / contactness_hoi.sum().clamp(min=1e-6)

        # skate: same feet, XY-L1 and height weighting as the metric
        feet_indices = torch.tensor([7, 8, 10, 11], dtype=torch.long, device=device)
        feet = sbj_joints[:, :, feet_indices]  # B x T x 4 x 3
        feet_xy_disp = (feet[:, 1:, :, :2] - feet[:, :-1, :, :2]).abs().sum(-1)
        with torch.no_grad():
            floor = sbj_joints[..., 2].amin(dim=(1, 2), keepdim=True)
            height = feet[:, 1:, :, 2] - floor
            h_thresh = torch.tensor([0.08, 0.08, 0.04, 0.04], device=device)
            weight = (height < h_thresh).float() * (2 - 2 ** (height / h_thresh))
        skating_loss = (weight * feet_xy_disp).sum() / weight.sum().clamp(min=1.0)

        return contacts_loss, skating_loss

    def get_guidance_update(
        self,
        mode,
        x_t,
        batch,
        t_motion,
        t_object,
        t_contact,
        condition_frame,
        condition_global,
        mask_hoi,
    ):
        B = x_t.shape[0]
        D_motion = self.data_motion_channels
        D_object = self.dim_output_object

        with torch.enable_grad():
            x_t_leaf = x_t.detach().requires_grad_(True)
            pred = self.denoising_model(
                x_t_leaf,
                t_motion.expand(B),
                t_object=t_object.expand(B),
                t_contact=t_contact.expand(B),
                condition_frame=condition_frame,
                condition_global=condition_global,
                mask_hoi=mask_hoi,
            )
            contacts_loss, skating_loss = self._guidance_losses(mode, pred, batch)
            grad_contact = torch.autograd.grad(
                contacts_loss, x_t_leaf, retain_graph=True, allow_unused=True
            )[0]
            grad_skate = torch.autograd.grad(skating_loss, x_t_leaf, allow_unused=True)[
                0
            ]

        update = torch.zeros_like(x_t)
        for grad, channels, scale, sampled in (
            (
                grad_contact,
                slice(D_motion, D_motion + D_object),
                self.cg_scale,
                mode[1] == "1",
            ),
            (
                grad_skate,
                slice(0, D_motion),
                self.cg_skate_relative_scale * self.cg_scale,
                mode[0] == "1",
            ),
        ):
            if grad is None or not sampled or scale == 0:
                continue
            g = grad[:, :, channels].flatten(1)
            g_norm = g.norm(dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = x_t[:, :, channels].flatten(1).norm(dim=1, keepdim=True)
            update[:, :, channels] = (-scale * x_norm * g / g_norm).view(
                update[:, :, channels].shape
            )

        return update

    @torch.no_grad()
    def _sample_step(
        self,
        mode: str,
        scheduler,
        x_t,
        condition_frame: Tensor,
        condition_global: Tensor,
        batch: BatchData,
        t_index: Tensor,
        extra_step_kwargs: dict,
        # Inference model
        inference_model: InferenceModelNaive,
        # Mask for HOI
        mask_hoi: Tensor,
    ):
        # Dimensions
        B, T, _ = x_t.shape
        D_motion = self.data_motion_channels
        D_object = self.dim_output_object
        D_contact = self.dim_contacts

        # Extract timesteps based on the index
        timesteps = scheduler.timesteps[t_index]

        t_motion = timesteps if mode[0] == "1" else torch.zeros_like(timesteps)
        t_object = timesteps if mode[1] == "1" else torch.zeros_like(timesteps)
        t_contact = timesteps if mode[2] == "1" else torch.zeros_like(timesteps)

        # Forward (pred is either noise or x_0)
        pred = self.denoising_model(
            x_t,
            t_motion.expand(B),
            t_object=t_object.expand(B),
            t_contact=t_contact.expand(B),
            condition_frame=condition_frame,
            condition_global=condition_global,
            mask_hoi=mask_hoi,
        )

        # Classifier guidance: computed on x_t, applied to x_{t-1} below.
        # Not applied at t == 0
        guidance_update = None
        if self.cg_apply:
            mask_t = torch.logical_and(timesteps <= self.cg_t_stamp, timesteps > 0)
            if mask_t.any():
                guidance_update = self.get_guidance_update(
                    mode,
                    x_t,
                    batch,
                    t_motion,
                    t_object,
                    t_contact,
                    condition_frame=condition_frame,
                    condition_global=condition_global,
                    mask_hoi=mask_hoi,
                )
                guidance_update = guidance_update * mask_t.reshape(-1, 1, 1).to(
                    guidance_update
                )

        # Make a step of inference model
        pred = inference_model.on_inference(pred, x_t, self.scheduler, timesteps)

        # Select part of the output based on the sampling mode
        _pred, _x_t = [], []
        x_t_motion, x_t_object, x_t_contact = torch.split(
            x_t, [D_motion, D_object, D_contact], dim=2
        )
        if mode[0] == "1":
            _pred.append(pred[:, :, :D_motion])
            _x_t.append(x_t_motion)
        if mode[1] == "1":
            _pred.append(pred[:, :, D_motion : D_motion + D_object])
            _x_t.append(x_t_object)
        if mode[2] == "1":
            _pred.append(
                pred[:, :, D_motion + D_object : D_motion + D_object + D_contact]
            )
            _x_t.append(x_t_contact)
        _pred = torch.cat(_pred, dim=2)
        _x_t = torch.cat(_x_t, dim=2)

        # flatten timesteps and pred with x_t
        _x_tm1 = scheduler.step(
            _pred.reshape(B * T, -1),
            timesteps,
            _x_t.reshape(B * T, -1),
            **extra_step_kwargs,
        ).prev_sample
        _x_tm1 = _x_tm1.reshape(B, T, -1)

        # assemble output from cond and prediction
        D_off = 0
        if mode[0] == "1":
            x_t_motion = _x_tm1[:, :, :D_motion]
            D_off += D_motion

        if mode[1] == "1":
            x_t_object = _x_tm1[:, :, D_off : D_off + D_object]
            D_off += D_object

        if mode[2] == "1":
            x_t_contact = _x_tm1[:, :, D_off:]

        x_tm1 = torch.cat([x_t_motion, x_t_object, x_t_contact], dim=2)

        if guidance_update is not None:
            x_tm1 = x_tm1 + guidance_update

        # Make a step of inference model
        x_tm1 = inference_model.on_scheduler(pred, x_tm1, self.scheduler)

        return x_tm1

    def forward_sample(
        self,
        # Sampling mode
        mode: str,
        # Data for conditioning
        batch: BatchData,
        # Inference model
        inference_model: InferenceModelNaive,
        # Diffusion scheduler
        scheduler: str | None = "ddpm",
        # Inference parameters
        num_inference_steps: int | None = 1000,
        eta: float | None = 0.0,  # for DDIM
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ) -> EgoModelOutput:
        """Naive inference per-batch without the conveyor"""
        # Set noise size
        B = 1 if batch is None else batch.batch_size()
        device = batch.body_pose_6d.device

        # Choose noise dimensionality
        D_motion = self.data_motion_channels
        D_object = self.dim_output_object
        D_contact = self.dim_contacts
        D = 0

        # sample noise and get conditioning
        x_motion_cond = torch.empty(0)
        if mode[0] == "1":
            x_t_motion = torch.randn(B, self.window_size, D_motion, device=device)
            D += D_motion
        else:
            # use motion as conditioning
            x_motion_cond = batch.body_pose_6d.to(device)
            x_t_motion = x_motion_cond.detach().clone()

        if mode[1] == "1":
            x_t_object = torch.randn(B, self.window_size, D_object, device=device)
            D += D_object
        else:
            # use object as conditioning
            x_object_cond = batch.obj_Rt_6d.to(device)
            x_t_object = x_object_cond.detach().clone()

        if mode[2] == "1":
            x_t_contact = torch.randn(B, self.window_size, D_contact, device=device)
            D += D_contact
        else:
            # use contact as conditioning
            x_contact_cond = batch.contacts.to(device)
            x_t_contact = x_contact_cond.detach().clone()

        if D == 0:
            raise NotImplementedError(f"Unknown forward mode: {mode}")
        x_t = torch.cat([x_t_motion, x_t_object, x_t_contact], dim=2)

        # Manage conditioning
        condition_frame, condition_global = self.get_conditioning(batch)

        # Setup scheduler
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = (
            self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        )
        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Prepare inference
        timestep_index = 0
        x_t, condition_frame, condition_global, timestep_index = (
            inference_model.before_inference(
                x_t,
                batch,
                self.scheduler,
                t_index=timestep_index,
                condition_frame=condition_frame,
                condition_global=condition_global,
            )
        )

        # Denoise
        progress_bar = tqdm(
            range(num_inference_steps),
            desc=f"Sampling {mode} ({B}, {D})",
            disable=disable_tqdm,
            ncols=80,
            leave=False,
        )
        for i, _ in enumerate(progress_bar):
            x_t, condition_frame, condition_global, timestep_index = (
                inference_model.before_step(
                    x_t,
                    batch,
                    t_index=timestep_index,
                    condition_frame=condition_frame,
                    condition_global=condition_global,
                )
            )

            x_t = self._sample_step(
                mode,
                scheduler,
                x_t,
                condition_frame,
                condition_global,
                batch,
                timestep_index,
                extra_step_kwargs,
                inference_model=inference_model,
                mask_hoi=~batch.is_fake_object,
            )

            timestep_index = inference_model.update_timestep(timestep_index)

        # Add result to the inference model
        sample_result: EgoModelOutput = inference_model.after_inference(batch, x_t)

        return sample_result

    def forward(
        self,
        batch: BatchData,
        mode="train",
        sample_type: str | None = None,
        **kwargs,
    ):
        """A wrapper around the forward method for training and inference"""
        if isinstance(
            batch, dict
        ):  # fixes a bug with multiprocessing where batch becomes a dict
            batch = BatchData(
                **batch
            )  # it really makes no sense, I do not understand it

        if mode == "train":
            return self.forward_train(batch, **kwargs)
        elif mode.startswith("sample"):
            return self.forward_sample(sample_type, batch, **kwargs)
        else:
            raise NotImplementedError(f"Unknown forward mode: {mode}")
