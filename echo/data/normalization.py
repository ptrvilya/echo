"""Normalization of the object pose representation.

Stats computed over the behave + omomo train splits for the head-relative representation
(model.object_pose == "relative").
"""

import torch

OBJ_TRANSL_MEAN = (-0.0271, -0.5957, 0.5025)
OBJ_TRANSL_STD = (0.2282, 0.3197, 0.3117)


def _stats(Rt_6d: torch.Tensor):
    mean = Rt_6d.new_tensor(OBJ_TRANSL_MEAN)
    std = Rt_6d.new_tensor(OBJ_TRANSL_STD)
    return mean, std


def normalize_obj_Rt_6d(Rt_6d: torch.Tensor) -> torch.Tensor:
    mean, std = _stats(Rt_6d)
    out = Rt_6d.clone()
    out[..., 6:] = (Rt_6d[..., 6:] - mean) / std
    return out


def denormalize_obj_Rt_6d(Rt_6d: torch.Tensor) -> torch.Tensor:
    mean, std = _stats(Rt_6d)
    out = Rt_6d.clone()
    out[..., 6:] = Rt_6d[..., 6:] * std + mean
    return out
