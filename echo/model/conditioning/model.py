from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from echo.data import BatchData


def fourier_encode(x, freqs):
    """Apply Fourier encoding to a tensor."""
    *batch_axes, x_dim = x.shape
    coeffs = 2.0 ** torch.arange(freqs, device=x.device)
    scaled = (x[..., None] * coeffs).reshape((*batch_axes, x_dim * freqs))
    return torch.cat(
        [
            x,
            torch.sin(torch.cat([scaled, scaled + torch.pi / 2.0], dim=-1)),
        ],
        dim=-1,
    )


class ConditioningModel(nn.Module):
    def __init__(
        self,
        window_size: int,
        use_hmd_feature_conditioning: bool = False,
        use_pointnext_conditioning: bool = False,
        use_class_conditioning: bool = False,
        dim_cond_hmd: int = 0,
        dim_cond_pointnext: int = 0,
        dim_cond_class: int = 0,
        condition_drop_prob: float = 0.1,
        fourier_encoding_freqs: int = 0,
        separate_fake_object: bool = False,
        use_prefix_conditioning: bool = False,
        prefix_prob: float = 0.5,
        prefix_len_max: int = 30,
        prefix_noise_std: float = 0.0,
        dim_prefix_motion: int = 0,
        dim_prefix_object: int = 0,
        dim_prefix_contacts_hoi: int = 0,
        dim_prefix_contacts_env: int = 0,
    ):
        super().__init__()
        self.window_size = window_size
        self.use_hmd_feature_conditioning = use_hmd_feature_conditioning
        self.use_pointnext_conditioning = use_pointnext_conditioning
        self.use_class_conditioning = use_class_conditioning
        self.condition_drop_prob = condition_drop_prob
        self.dim_cond_class = dim_cond_class
        self.dim_cond_pointnext = dim_cond_pointnext
        self.fourier_encoding_freqs = fourier_encoding_freqs
        self.separate_fake_object = separate_fake_object
        self.use_prefix_conditioning = use_prefix_conditioning
        self.prefix_prob = prefix_prob
        self.prefix_len_max = prefix_len_max
        self.prefix_noise_std = prefix_noise_std
        # prefix layout matches x_t: [motion, object, contacts_hoi, contacts_env]
        self.dim_prefix = (
            dim_prefix_motion
            + dim_prefix_object
            + dim_prefix_contacts_hoi
            + dim_prefix_contacts_env
        )
        self.prefix_hoi_slice = slice(
            dim_prefix_motion,
            dim_prefix_motion + dim_prefix_object + dim_prefix_contacts_hoi,
        )

        # Additional input dimensions for conditioning
        self.cond_frame_channels = 0
        self.cond_global_channels = 0
        if self.use_prefix_conditioning:
            self.cond_frame_channels += self.dim_prefix + 1  # + known-frame mask
        if self.use_hmd_feature_conditioning:
            self.cond_frame_channels += dim_cond_hmd

            if self.fourier_encoding_freqs > 0:
                self.cond_frame_channels += (
                    dim_cond_hmd * 2 * self.fourier_encoding_freqs
                )

        if self.use_pointnext_conditioning:
            self.projection_condition_pointnext = nn.Linear(
                dim_cond_pointnext, dim_cond_pointnext // 2, bias=False
            )
            self.null_pointnext = nn.Parameter(
                torch.zeros(dim_cond_pointnext), requires_grad=self.separate_fake_object
            )

            self.cond_global_channels += dim_cond_pointnext // 2
        else:
            self.projection_condition_pointnext = None

        if self.use_class_conditioning:
            self.cond_global_channels += dim_cond_class

    def init_weights(self):
        pass

    def sample_prefix(self, x_0: torch.Tensor):
        """Training prefix: clean x_0 of the first K frames for a random subset
        of samples. Returns (prefix_x0, prefix_mask) or (None, None)."""
        if not self.use_prefix_conditioning:
            return None, None
        B, T, _ = x_0.shape
        device = x_0.device
        K = torch.randint(1, self.prefix_len_max + 1, (B, 1), device=device)
        use = torch.rand(B, 1, device=device) < self.prefix_prob
        mask = (torch.arange(T, device=device).view(1, T) < K) & use
        mask = mask.unsqueeze(-1).float()

        prefix = x_0.detach()
        if self.prefix_noise_std > 0:
            prefix = prefix + self.prefix_noise_std * torch.randn_like(prefix)
        return prefix * mask, mask

    def get_conditioning(
        self,
        batch: BatchData,
        prefix_x0: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ):
        B = batch.batch_size()
        T = self.window_size
        condition_frame, condition_global = [], []

        if self.use_hmd_feature_conditioning:
            hmd_feature = batch.hmd_condition

            if self.fourier_encoding_freqs > 0:
                hmd_feature = fourier_encode(
                    hmd_feature, freqs=self.fourier_encoding_freqs
                )

            condition_frame.append(hmd_feature.reshape(B, T, -1))

        if self.use_prefix_conditioning:
            device = batch.hmd_condition.device
            if prefix_x0 is None:
                prefix_x0 = torch.zeros(B, T, self.dim_prefix, device=device)
                prefix_mask = torch.zeros(B, T, 1, device=device)
            elif batch.is_fake_object is not None:
                # no object/HOI-contact information for motion-only samples
                prefix_x0 = prefix_x0.clone()
                fake = batch.is_fake_object.view(B, 1, 1).to(device)
                prefix_x0[:, :, self.prefix_hoi_slice] *= (~fake).float()
            condition_frame.append(prefix_x0)
            condition_frame.append(prefix_mask)

        if self.use_pointnext_conditioning:
            mask_hoi = ~batch.is_fake_object.view(B, 1).expand(
                B, self.dim_cond_pointnext
            )
            if self.separate_fake_object:
                null_pointnext = self.null_pointnext.view(1, -1).expand(B, -1)
                pointnext = torch.where(mask_hoi, batch.obj_pointnext, null_pointnext)
            else:
                pointnext = batch.obj_pointnext

            condition_global.append(self.projection_condition_pointnext(pointnext))

        if self.use_class_conditioning:
            obj_class = batch.obj_class_id

            obj_class_one_hot = F.one_hot(
                obj_class, num_classes=self.dim_cond_class
            ).float()
            obj_class_one_hot = obj_class_one_hot.reshape(
                B, self.dim_cond_class
            )  # B x num_classes
            obj_class_one_hot = obj_class_one_hot.to(batch.obj_Rt_6d.device)

            condition_global.append(obj_class_one_hot)

        # stack across feature dimension
        if len(condition_frame) > 0:
            condition_frame = torch.cat(condition_frame, dim=2)
        else:
            condition_frame = torch.empty(0)
        if len(condition_global) > 0:
            condition_global = torch.cat(condition_global, dim=1)
        else:
            condition_global = torch.empty(0)

        # dropping conditioning for regularization, independently per sample
        if self.training and self.condition_drop_prob > 0:
            device = (
                condition_frame.device
                if condition_frame.numel() > 0
                else condition_global.device
            )
            drop = torch.rand(B, device=device) < self.condition_drop_prob
            if condition_frame.numel() > 0:
                condition_frame = torch.where(
                    drop.view(B, 1, 1),
                    torch.zeros_like(condition_frame),
                    condition_frame,
                )
            if condition_global.numel() > 0:
                condition_global = torch.where(
                    drop.view(B, 1),
                    torch.zeros_like(condition_global),
                    condition_global,
                )

        return condition_frame, condition_global
