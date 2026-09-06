"""
https://github.com/milmor/diffusion-transformer/blob/main/dit.py
"""

import torch
import torch.nn as nn

from .common import DiTBlock, get_timestep_embedding


class ModalityProjection(nn.Module):
    def __init__(
        self, dim_input: int, dim_intermediate_embdedding: int, dim_embedding: int
    ):
        super(ModalityProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim_input, dim_intermediate_embdedding, bias=False),
            nn.LayerNorm(dim_intermediate_embdedding),
            nn.SiLU(),
            nn.Linear(dim_intermediate_embdedding, dim_embedding, bias=True),
        )

    def forward(self, x):
        return self.projection(x)


class HOICXDiT(nn.Module):
    "Cross attention version of DiT"

    def __init__(
        self,
        dim_motion_feature: int,  # e.g. 6d rotation
        dim_intermediate_embedding: int,
        dim_condition_frame: int,  # e.g. 6d rotation + velocity + position
        dim_condition_global: int,  # e.g. obj_class + obj_pointnext
        dim_output_motion: int,  # e.g. 6d rotation
        dim_contacts: int,
        dim_contacts_env: int,
        dim_contacts_hoi: int,
        dim_output_object: int,
        dim_embedding: int = 512,
        dim_feedforward: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        num_joints: int = 21,
        dropout_attn: float = 0.1,
        dropout_mlp: float = 0.1,
        window_size: int = 60,
    ):
        super(HOICXDiT, self).__init__()
        # variables
        self.dim_input_motion = dim_motion_feature * num_joints
        self.num_joints = num_joints
        self.dim_embedding = dim_embedding
        self.dim_motion_feature = dim_motion_feature
        self.dim_output_motion = dim_output_motion
        self.dim_output_object = dim_output_object
        self.dim_contacts = dim_contacts
        self.dim_contacts_env = dim_contacts_env
        self.dim_contacts_hoi = dim_contacts_hoi
        self.dim_intermediate_embedding = dim_intermediate_embedding
        self.dim_condition_frame = dim_condition_frame
        self.dim_condition_global = dim_condition_global

        # embedders
        self.projection_H = ModalityProjection(
            self.dim_input_motion,
            self.dim_intermediate_embedding,
            self.dim_intermediate_embedding,
        )
        self.projection_O = ModalityProjection(
            self.dim_output_object,
            self.dim_intermediate_embedding,
            self.dim_intermediate_embedding,
        )
        self.projection_C_hoi = ModalityProjection(
            self.dim_contacts_hoi,
            self.dim_intermediate_embedding // 2,
            self.dim_intermediate_embedding // 2,
        )
        self.projection_C_env = ModalityProjection(
            self.dim_contacts_env,
            self.dim_intermediate_embedding // 2,
            self.dim_intermediate_embedding // 2,
        )
        self.projection_Cond_frame = ModalityProjection(
            self.dim_condition_frame,
            self.dim_embedding,
            self.dim_embedding,
        )
        self.projection_Cond_global = nn.Linear(
            self.dim_condition_global, self.dim_embedding // 2, bias=False
        )
        self.projection_fusion = nn.Sequential(
            nn.LayerNorm(3 * self.dim_intermediate_embedding),
            nn.Linear(
                3 * self.dim_intermediate_embedding,
                self.dim_embedding,
                bias=False,
            ),
            nn.LayerNorm(self.dim_embedding),
            nn.SiLU(),
            nn.Linear(
                self.dim_embedding,
                self.dim_embedding,
                bias=True,
            ),
        )
        self.projection_T = nn.Sequential(
            nn.Linear(
                3 * self.dim_intermediate_embedding, self.dim_embedding // 2, bias=True
            ),
            nn.SiLU(),
            nn.Linear(self.dim_embedding // 2, self.dim_embedding // 2, bias=True),
        )

        # Positional and modality embedding
        self.emb_null_object = nn.Parameter(
            torch.randn(self.dim_intermediate_embedding),
            requires_grad=True,
        )
        self.emb_null_contact = nn.Parameter(
            torch.randn(self.dim_intermediate_embedding // 2),
            requires_grad=True,
        )

        # main blocks
        self.transformer_enc = nn.ModuleList(
            [
                DiTBlock(
                    dim_input=dim_embedding,
                    num_heads=num_heads,
                    dim_hidden=dim_feedforward,
                    dropout_attn=dropout_attn,
                    dropout_mlp=dropout_mlp,
                    use_cross_attn=False,
                    use_rope=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.transformer_dec = nn.ModuleList(
            [
                DiTBlock(
                    dim_input=dim_embedding,
                    num_heads=num_heads,
                    dim_hidden=dim_feedforward,
                    dropout_attn=dropout_attn,
                    dropout_mlp=dropout_mlp,
                    use_cross_attn=True,
                    use_rope=True,
                )
                for _ in range(num_layers)
            ]
        )

        # decoders
        self.decoder_H = nn.ModuleDict(
            {
                "projection": nn.Sequential(
                    nn.Linear(self.dim_embedding, self.dim_input_motion, bias=False),
                    nn.LayerNorm(self.dim_input_motion),
                    nn.SiLU(),
                ),
                "decoder": nn.Linear(
                    self.dim_motion_feature, self.dim_output_motion, bias=True
                ),
            }
        )
        self.decoder_O = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_intermediate_embedding, bias=False),
            nn.LayerNorm(self.dim_intermediate_embedding),
            nn.SiLU(),
            nn.Linear(
                self.dim_intermediate_embedding, self.dim_output_object, bias=True
            ),
        )
        self.decoder_C = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_intermediate_embedding, bias=False),
            nn.LayerNorm(self.dim_intermediate_embedding),
            nn.SiLU(),
            nn.Linear(self.dim_intermediate_embedding, self.dim_contacts, bias=True),
        )

    def init_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_enc:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        for block in self.transformer_dec:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def prepare_time(self, t_motion, t_object, t_contact, device):
        # Embed and project timesteps
        t_emb_motion = get_timestep_embedding(
            self.dim_intermediate_embedding, t_motion.reshape(-1), device
        )
        t_emb_object = get_timestep_embedding(
            self.dim_intermediate_embedding, t_object.reshape(-1), device
        )
        t_emb_contact = get_timestep_embedding(
            self.dim_intermediate_embedding, t_contact.reshape(-1), device
        )

        t_emb = torch.cat([t_emb_motion, t_emb_object, t_emb_contact], dim=-1)
        t_emb = self.projection_T(t_emb)
        return t_emb

    def prepare_token(self, x_input, emb_cond_frame, mask_hoi):
        B, T, F = x_input.shape  # batch x frames x joints * features

        x_motion, x_object, x_contact = x_input.split(
            [self.dim_input_motion, self.dim_output_object, self.dim_contacts], dim=2
        )

        # For object and contacts overwrite HOI parts for motion only sequences with null tokens
        # For contacts only overwrite the HOI contact dimensions
        emb_object_real = self.projection_O(x_object)
        emb_contact_hoi_real = self.projection_C_hoi(
            x_contact[:, :, : self.dim_contacts_hoi]
        )

        emb_object_null = self.emb_null_object.view(1, 1, -1).expand(B, T, -1)
        emb_contact_null = self.emb_null_contact.view(1, 1, -1).expand(B, T, -1)
        mask = mask_hoi.view(B, 1, 1)
        emb_object = torch.where(mask, emb_object_real, emb_object_null)
        emb_contact_hoi = torch.where(mask, emb_contact_hoi_real, emb_contact_null)

        # Project, then fuse
        emb_motion = self.projection_H(x_motion)
        emb_contact_env = self.projection_C_env(
            x_contact[:, :, self.dim_contacts_hoi :]
        )
        emb_contact = torch.cat([emb_contact_hoi, emb_contact_env], dim=2)
        emb_cond = self.projection_Cond_frame(emb_cond_frame)

        emb_x = self.projection_fusion(
            torch.cat([emb_motion, emb_object, emb_contact], dim=2)
        )
        return emb_x, emb_cond

    def unembed_prediction(self, output):
        B, T, F = output.shape

        # motion
        motion = self.decoder_H["projection"](output)
        motion = motion.reshape(B, T, self.num_joints, self.dim_motion_feature)
        motion = self.decoder_H["decoder"](motion)
        motion = motion.reshape(B, T, -1)  # B x T x J * F

        # object & contacts
        object = self.decoder_O(output)
        contacts = self.decoder_C(output)

        output = torch.cat([motion, object, contacts], dim=-1)

        return output

    def forward(
        self,
        x,
        t_x,
        emb_cond_frame,
        emb_cond_global,
        t_object,
        mask_hoi,
        t_contact=None,
    ):
        emb_token, emb_cond = self.prepare_token(x, emb_cond_frame, mask_hoi)
        emb_t = self.prepare_time(
            t_x.to(x.device), t_object.to(x.device), t_contact.to(x.device), x.device
        )

        global_cond = self.projection_Cond_global(emb_cond_global)
        global_cond = torch.cat([emb_t, global_cond], dim=1)

        for block in self.transformer_enc:
            emb_cond = block(emb_cond, global_cond, None)

        _x = emb_token
        for block in self.transformer_dec:
            _x = block(_x, global_cond, cross_attn_cond=emb_cond)

        out = self.unembed_prediction(_x)
        return out
