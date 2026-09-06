import contextlib
from typing import Optional

import torch
from torch import Tensor, nn

from .transformer.xdit_hoic import HOICXDiT


class DenoisingModel(nn.Module):
    """Model that denoises parameters at each diffusion step"""

    def __init__(
        self,
        name: str,
        dim_motion_feature: int,
        dim_intermediate_embedding: int,
        dim_output_motion: int,
        dim_condition_frame: int,
        dim_condition_global: int,
        num_joints: int,
        window_size: int,
        dim_input_object: Optional[int] = None,
        dim_output_object: Optional[int] = None,
        dim_contacts: Optional[int] = None,
        dim_contacts_env: Optional[int] = None,
        dim_contacts_hoi: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.name = name
        # set to torch.bfloat16 (via train.mixed_precision) to run the
        # transformer in mixed precision; inputs/outputs stay fp32
        self.autocast_dtype: Optional[torch.dtype] = None

        base_kwargs = dict(
            dim_motion_feature=dim_motion_feature,
            dim_intermediate_embedding=dim_intermediate_embedding,
            dim_output_motion=dim_output_motion,
            dim_condition_frame=dim_condition_frame,
            dim_condition_global=dim_condition_global,
            num_joints=num_joints,
            window_size=window_size,
        )

        if self.name != "hoic_xdit":
            raise ValueError(f"Unknown denoising model {self.name}")
        self.model = HOICXDiT(
            **base_kwargs,
            dim_output_object=dim_output_object,
            dim_contacts_env=dim_contacts_env,
            dim_contacts_hoi=dim_contacts_hoi,
            dim_contacts=dim_contacts,
            **kwargs,
        )

    def init_weights(self):
        self.model.init_weights()

    def forward(
        self,
        inputs: Tensor,
        t: Tensor,
        condition_frame: Optional[Tensor] = None,
        t_object: Optional[Tensor] = None,
        t_contact: Optional[Tensor] = None,
        condition_global: Optional[Tensor] = None,
        mask_hoi: Optional[Tensor] = None,
    ) -> Tensor:
        """Receives input of shape (B, in_channels) and returns output
        of shape (B, out_channels)"""
        with self._autocast():
            out = self.model(
                inputs,
                t,
                emb_cond_frame=condition_frame,
                t_object=t_object,
                t_contact=t_contact,
                emb_cond_global=condition_global,
                mask_hoi=mask_hoi,
            )
        # everything outside the transformer (losses, SE3 ops) stays fp32
        return out.float()

    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=self.autocast_dtype)
