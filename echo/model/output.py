from dataclasses import dataclass, fields
from logging import getLogger
from typing import Any

import torch
from torch import Tensor

from echo.data.batch_data import BatchData
from echo.data.normalization import denormalize_obj_Rt_6d, normalize_obj_Rt_6d
from echo.utils.rotation import SE3_to_Rt_6d, rotation_6d_to_matrix
from echo.utils.transforms import SE3, SO3

logger = getLogger(__name__)


@dataclass
class EgoModelOutput:
    # SMPL parameters
    # betas come from batch
    # transl and global_orient are recovered from conditioning
    betas: Tensor | None = None
    transl: Tensor | None = None
    global_orient: Tensor | None = None

    # model predictions
    body_pose_6d: Tensor | None = None
    obj_Rt_6d: Tensor | None = None  # either same as T_world_obj or w.r.t. Head Frame
    diff_contacts: Tensor | None = None

    sbj_joints: Tensor | None = None
    sbj_vertices: Tensor | None = None
    obj_keypoints: Tensor | None = None

    # Global transformation of the object
    T_world_obj: Tensor | None = None

    # Conditioning
    T_can_world: Tensor | None = None
    T_world_head: Tensor | None = None
    obj_class_id: Tensor | None = None

    # Metadata
    real_slice_len: Tensor | None = None
    timesteps_motion: Tensor | None = None
    timesteps_object: Tensor | None = None
    timesteps_contact: Tensor | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def is_empty(self) -> bool:
        is_empty = True
        for field in fields(self):
            attr = getattr(self, field.name, None)
            if attr is not None:
                is_empty = False
                break
        return is_empty

    @staticmethod
    def convert_object_from_global(
        T_world_obj: Tensor,
        T_world_head: Tensor,
        T_can_world: Tensor,
        target_object_pose: str,
    ) -> Tensor:
        if target_object_pose == "global":
            # obj_Rt_6d == T_world_obj
            obj_Rt_6d = SE3_to_Rt_6d(T_world_obj)
        elif target_object_pose == "canonical":
            # obj_Rt_6d == T_can_world @ T_world_obj
            T_world_obj = SE3(T_world_obj)
            T_can_world = SE3(T_can_world)

            T_can_obj = T_can_world @ T_world_obj
            obj_Rt_6d = SE3_to_Rt_6d(T_can_obj)

            T_can_world = T_can_world.parameters()
            T_world_obj = T_world_obj.parameters()
        elif target_object_pose == "relative":
            # obj_Rt_6d == T_world_head.inverse() @ T_world_obj
            T_world_obj = SE3(T_world_obj)
            T_world_head = SE3(T_world_head)

            T_head_obj = T_world_head.inverse() @ T_world_obj
            obj_Rt_6d = SE3_to_Rt_6d(T_head_obj)

            T_world_head = T_world_head.parameters()
            T_world_obj = T_world_obj.parameters()
        else:
            raise ValueError(f"Unknown object_pose: {target_object_pose}")

        return normalize_obj_Rt_6d(obj_Rt_6d)

    @staticmethod
    def convert_object_to_global(
        obj_Rt_6d: Tensor,
        T_world_head: Tensor,
        T_can_world: Tensor,
        source_object_pose: str,
    ):
        obj_R, obj_t = torch.split(denormalize_obj_Rt_6d(obj_Rt_6d), [6, 3], dim=-1)
        obj_R = rotation_6d_to_matrix(obj_R)

        if source_object_pose == "global":
            T_world_obj = SE3.from_rotation_and_translation(
                SO3.from_matrix(obj_R), obj_t
            ).parameters()
        elif source_object_pose == "canonical":
            T_can_obj = SE3.from_rotation_and_translation(SO3.from_matrix(obj_R), obj_t)

            T_can_world = SE3(T_can_world)
            T_world_obj = T_can_world.inverse() @ T_can_obj

            T_can_world = T_can_world.parameters()
            T_world_obj = T_world_obj.parameters()
        elif source_object_pose == "relative":
            T_head_obj = SE3.from_rotation_and_translation(
                SO3.from_matrix(obj_R), obj_t
            )
            T_world_head = SE3(T_world_head)

            T_world_obj = (T_world_head @ T_head_obj).parameters()
            T_world_head = T_world_head.parameters()
        else:
            raise ValueError(f"Unknown object_pose: {source_object_pose}")

        return T_world_obj

    @classmethod
    def unpack(
        self,
        output: Tensor,
        timesteps_motion: Tensor | None = None,
        timesteps_object: Tensor | None = None,
        timesteps_contact: Tensor | None = None,
        batch: BatchData = None,
        object_pose: str | None = "canonical",
    ) -> "EgoModelOutput":
        NUM_JOINTS = 21
        # parse network output vector

        betas = batch.betas if batch is not None else None
        T_world_head = batch.T_world_head if batch is not None else None
        T_can_world = batch.T_can_world if batch is not None else None
        obj_class_id = batch.obj_class_id if batch is not None else None
        real_slice_len = batch.real_slice_len if batch is not None else None

        if output.shape[-1] > NUM_JOINTS * 6:
            obj_Rt_6d = output[..., NUM_JOINTS * 6 : NUM_JOINTS * 6 + 9]

            if (
                T_world_head is not None
                and T_can_world is not None
                and object_pose != "dummy"
            ):
                T_world_obj = self.convert_object_to_global(
                    obj_Rt_6d,
                    T_world_head,
                    T_can_world,
                    object_pose,
                )
            else:
                T_world_obj = None
                obj_Rt_6d = None
        else:
            obj_Rt_6d = None
            T_world_obj = None

        if output.shape[-1] > NUM_JOINTS * 6 + 9:
            contacts = output[..., NUM_JOINTS * 6 + 9 :]
        else:
            contacts = None

        # Hack needed for val loss computation
        if timesteps_object is None:
            timesteps_object = timesteps_motion
        if timesteps_contact is None:
            timesteps_contact = timesteps_motion

        return EgoModelOutput(
            body_pose_6d=output[..., : NUM_JOINTS * 6],
            obj_Rt_6d=obj_Rt_6d,
            diff_contacts=contacts,
            timesteps_motion=timesteps_motion,
            timesteps_object=timesteps_object,
            timesteps_contact=timesteps_contact,
            betas=betas,
            T_world_head=T_world_head,
            T_can_world=T_can_world,
            real_slice_len=real_slice_len,
            T_world_obj=T_world_obj,
            obj_class_id=obj_class_id,
        )

    def __len__(self):
        for field in fields(self):
            attr = getattr(self, field.name, None)
            if attr is not None:
                return len(attr)
