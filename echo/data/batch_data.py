from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

import torch
from torch import Tensor


@dataclass
class BatchData:
    # Body model params (pose in aa format)
    betas: Optional[Tensor] = None
    global_orient: Optional[Tensor] = None
    body_pose: Optional[Tensor] = None
    transl: Optional[Tensor] = None
    left_hand_pose: Optional[Tensor] = None
    right_hand_pose: Optional[Tensor] = None

    # Contacts
    contacts: Optional[Tensor] = None

    # Motion features
    hmd_condition: Optional[Tensor] = None
    T_world_head: Optional[Tensor] = None
    body_pose_6d: Optional[Tensor] = None

    # Object
    is_fake_object: Optional[Tensor] = None
    include_object: Optional[Tensor] = None
    T_world_obj: Optional[Tensor] = None
    obj_Rt_6d: Optional[Tensor] = None
    obj_class_id: Optional[Tensor] = None
    obj_pointnext: Optional[Tensor] = None

    # Vertices & joints
    sbj_vertices: Optional[Tensor] = None
    obj_keypoints: Optional[Tensor] = None
    sbj_joints: Optional[Tensor] = None

    # Canonicalization transform
    T_can_world: Optional[Tensor] = None
    T_can_head: Optional[Tensor] = None

    # Sub-sequence information
    t_start: Optional[Tensor] = None
    t_end: Optional[Tensor] = None
    seq_name: Optional[Union[str, List[str]]] = None
    seq_id: Optional[Union[int, List[int]]] = None
    real_slice_len: Optional[Union[int, List[int]]] = None
    total_sequence_len: Optional[Union[int, List[int]]] = None
    is_end_of_sequence: Optional[Union[bool, List[bool]]] = None

    # Additional information
    # 0 - neutral, 1- female, 2 - male
    timestamps: Optional[Tensor] = None
    timestamps_obj: Optional[Tensor] = None
    sbj_gender: Optional[Union[int, List[int]]] = None
    obj_class_name: Optional[Union[str, List[str]]] = None
    meta: dict = field(default_factory=lambda: {})

    def to(self, *args, **kwargs):
        new_params = {}
        for field_name in iter(self):
            value = getattr(self, field_name)
            if isinstance(value, (torch.Tensor)):
                new_params[field_name] = value.to(*args, **kwargs)
            else:
                new_params[field_name] = value
        batch_data = type(self)(**new_params)
        return batch_data

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def batch_size(self):
        for f in iter(self):
            if f != "meta":
                attr = self.__getattribute__(f)
                if attr is not None:
                    return len(attr)

    # the following functions make sure **batch_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            if f.name.startswith("_"):
                continue

            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return sum(1 for f in iter(self))

    @classmethod
    def concatenate(cls, batches: List["BatchData"]) -> "BatchData":
        """
        Concatenates a list of `BatchData` objects into a single `BatchData` object.
        """
        collated = defaultdict(list)
        for batch in batches:
            for f in fields(batch):
                if not f.init:
                    continue

                collated[f.name].append(getattr(batch, f.name))

        for k, v in collated.items():
            if len(v) > 0 and isinstance(v[0], torch.Tensor):
                collated[k] = torch.cat(v, dim=min(v[0].ndim - 1, 1))
            else:
                collated[k] = v

        return BatchData(**collated)

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """
        elem = batch[0]

        if isinstance(elem, cls):
            collated = {}

            # Get all fields once
            elem_fields = fields(elem)

            # Collect values for each field
            for f in elem_fields:
                if not f.init:
                    continue

                list_values = []
                for d in batch:
                    attr = getattr(d, f.name)
                    if attr is None:
                        list_values = None
                        break
                    list_values.append(attr)

                # Check if all values are not None
                if list_values is not None and len(list_values) > 0:
                    # Check if they're tensors - avoid recursive call for tensors
                    if isinstance(list_values[0], torch.Tensor):
                        collated[f.name] = torch.stack(list_values, dim=0)
                    else:
                        # Recursive call for nested structures
                        collated[f.name] = cls.collate(list_values)
                else:
                    collated[f.name] = None

            return cls(**collated)
        else:
            return torch.utils.data.default_collate(batch)
