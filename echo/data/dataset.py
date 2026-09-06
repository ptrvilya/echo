import pickle as pkl
import random
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from echo.configs import ConditioningModelConfig, EgoHOIModelConfig
from echo.data_parsers.common import (
    get_reset_Z_rotation,
)
from echo.utils.contacts import compute_env_contactness, compute_hoi_contactness
from echo.utils.rotation import (
    matrix_to_rotation_6d,
)
from echo.data.normalization import normalize_obj_Rt_6d
from echo.utils.training import simulate_observation_noise
from echo.utils.transforms import SE3, SO3

from .batch_data import BatchData

logger = getLogger(__name__)


class H5DataSample(NamedTuple):
    sequence: str
    sequence_id: int
    sequence_len: int
    slice_id: int
    slice_len: int
    is_padded: bool
    is_last: bool


@dataclass
class MotionDataset:
    name: str
    root: Path
    split: str
    # Model configurations
    conditioning_config: ConditioningModelConfig
    model_config: EgoHOIModelConfig
    # Sampling configuration
    include_object: bool = False
    include_fake_object: bool = False
    h5dataset_folder: Optional[Path] = None
    assets_folder: Optional[Path] = None
    split_file: Optional[str] = None
    slice_strategy: str = "random"
    overlap_size: int = 0  # only used for deterministic_w_overlap
    noise_simulation_percent: int = 0
    # metadata
    num_betas: int = 10
    obj2classid: Optional[Dict[str, int]] = None
    fps: Optional[int] = 30
    preload_files: bool = False
    # internal fields
    _head_id: int = 15  # head joint id in SMPL model
    _num_downsampled_verts: Optional[int] = None
    h5datasets: List[Union[Dict, h5py.File]] = field(init=False)

    def __post_init__(self) -> None:
        # sanity check
        if self.include_object:
            assert self.obj2classid is not None, (
                "Object class id mapping is required when object is included"
            )
            self.obj_keypoints = self._load_canonical_obj_keypoints()
        else:
            self.obj2classid = dict()
            self.obj_keypoints = dict()

        # Save param for convenience
        self.window_size = self.model_config.window_size
        self._include_hoi_contact = self.model_config.include_hoi_contact
        self._include_env_contact = self.model_config.include_env_contact
        self._separate_fake_object = self.conditioning_config.separate_fake_object

        # Open h5 dataset
        self.h5dataset_folder = self.root / f"dataset_{self.split}_{self.fps}fps"
        self.sequences = sorted(
            [sequence.stem for sequence in self.h5dataset_folder.glob("*.hdf5")]
        )
        if self.preload_files:
            self.h5datasets = []
            for sequence in tqdm(self.sequences, ncols=80, desc="Preloading h5 files"):
                with h5py.File(
                    self.h5dataset_folder / f"{sequence}.hdf5", "r"
                ) as h5file:
                    attrs = {"T": h5file.attrs["T"], "gender": h5file.attrs["gender"]}

                    h5_dict = dict()

                    # SMPL params and Head+Hands features
                    for key in [
                        "betas",
                        "transl",
                        "global_orient",
                        "body_pose",
                        "T_world_head",
                        "T_world_hands",
                        "T_head_tm1_head_t",
                        "T_hands_tm1_hands_t",
                        "smpl_joints_world",
                        "sparse_R_6d_global",
                        "sparse_XYZ_global",
                        "sparse_w_global",
                        "sparse_V_global",
                        "body_pose_6d",
                        "hand_pose_6d",
                        "timestamps",
                    ]:
                        h5_dict[key] = torch.from_numpy(h5file[key][:]).float()

                    if self.include_object:
                        for key in [
                            "obj_R",
                            "obj_transl",
                            # "timestamps_obj",
                        ]:
                            h5_dict[key] = torch.from_numpy(h5file[key][:]).float()

                        attrs["object_name"] = h5file.attrs["object_name"]

                        if self._include_hoi_contact:
                            h5_dict["hoi_contacts"] = torch.from_numpy(
                                h5file["hoi_contacts"][:]
                            ).float()

                    if self._include_env_contact:
                        h5_dict["env_contacts"] = torch.from_numpy(
                            h5file["env_contacts"][:]
                        ).float()

                    h5_dict["attrs"] = attrs
                    self.h5datasets.append(h5_dict)
        else:
            self.h5datasets = [
                h5py.File(self.h5dataset_folder / f"{sequence}.hdf5", "r")
                for sequence in self.sequences
            ]

        # Load H5DataSample and sort by timestamp
        self.data: List[H5DataSample] = self._create_samples()

        # Load pointnext object encoding
        if (
            self.conditioning_config.use_pointnext_conditioning
            and not self.include_fake_object
        ):
            assert self.obj2classid is not None, (
                "Object class id mapping is required when object is included"
            )
            self.obj_pointnext = self._load_object_pointnext()
        else:
            self.obj_pointnext = dict()

        # Load downsampled vertices
        if self._include_hoi_contact:
            assert self.assets_folder is not None
            self._num_downsampled_verts = len(
                np.load(self.assets_folder / "smplx_downsample_indices.npy")
            )

        # Log info
        logger.info(self.__str__())

    def _create_samples(self):
        # load sequence list filtering by window size
        data = []

        total_T = 0
        num_sequences = 0
        for seq_id, seq in enumerate(self.sequences):
            T: int = (
                self.h5datasets[seq_id]["attrs"]["T"]
                if self.preload_files
                else self.h5datasets[seq_id].attrs["T"]
            )

            if self.slice_strategy in ["deterministic", "deterministic_w_overlap"]:
                # prepare for batching
                total_T += T
                num_sequences += 1

                # Determine step_size
                if self.slice_strategy == "deterministic_w_overlap":
                    step_size = self.window_size - self.overlap_size
                else:
                    step_size = self.window_size

                if T <= self.window_size:
                    num_slices = 1
                else:
                    num_slices = (T - self.window_size + step_size - 1) // step_size + 1

                for slice_id in range(num_slices):
                    slice_start = slice_id * step_size
                    slice_end = slice_start + self.window_size

                    # Check if this slice exceeds the sequence length
                    if slice_end > T:
                        slice_len = T - slice_start
                        slice_end = T
                        is_padded = True
                    else:
                        slice_len = self.window_size
                        is_padded = False

                    data.append(
                        H5DataSample(
                            sequence=seq,
                            sequence_id=num_sequences - 1,
                            sequence_len=T,
                            slice_id=slice_id,
                            slice_len=slice_len,
                            is_padded=is_padded,
                            is_last=slice_id == num_slices - 1,
                        )
                    )
            elif self.slice_strategy == "random":
                # Potentially repeat num_slices times for a
                # proportional sampling of sequence_id
                total_T += T
                num_sequences += 1
                data.append(
                    H5DataSample(
                        sequence=seq,
                        sequence_id=num_sequences - 1,
                        sequence_len=T,
                        slice_id=-1,
                        slice_len=min(self.window_size, T),
                        is_padded=T < self.window_size,
                        is_last=False,
                    )
                )
            else:
                raise ValueError(f"Unknown slice strategy {self.slice_strategy}")

        self.num_sequences = num_sequences
        if self.slice_strategy in ["deterministic", "deterministic_w_overlap"]:
            self._data_length = len(data)
        elif self.slice_strategy == "random":
            self._data_length = total_T // self.window_size

        return data

    def _load_canonical_obj_keypoints(self):
        assert self.obj2classid is not None
        self.classid2obj = {v: k for k, v in self.obj2classid.items()}

        canonical_obj_keypoints = dict()
        for class_id, object_name in self.classid2obj.items():
            canonical_obj_keypoints[class_id] = dict(
                np.load(str(self.root / "object_keypoints" / f"{object_name}.npz"))
            )
        return canonical_obj_keypoints

    def _load_object_pointnext(self):
        assert self.obj2classid is not None
        with (self.root / "object_pointnext.pkl").open("rb") as fp:
            _obj_pointnext = pkl.load(fp)

        obj_pointnext = {
            v: torch.tensor(_obj_pointnext[k]).float()
            for k, v in self.obj2classid.items()
        }

        return obj_pointnext

    def __str__(self) -> str:
        return f"MotionDataset {self.name}: split={self.split} #data samples={self._data_length}"

    def __len__(self) -> int:
        # approximate number of slices
        return self._data_length

    def _get_tensor(
        self, seq_id: int, key: str, start_t: int, end_t: int
    ) -> torch.Tensor:
        data = self.h5datasets[seq_id][key][start_t:end_t]
        if self.preload_files:
            return data.clone()
        else:
            return torch.from_numpy(data).float()

    def __getitem__(self, idx: int) -> BatchData:
        if self.slice_strategy == "deterministic":
            data_sample = self.data[idx]
            seq_id = data_sample.sequence_id

            T = data_sample.sequence_len
            start_t = data_sample.slice_id * self.window_size
            end_t = start_t + data_sample.slice_len
        elif self.slice_strategy == "deterministic_w_overlap":
            data_sample = self.data[idx]
            seq_id = data_sample.sequence_id

            T = data_sample.sequence_len
            start_t = data_sample.slice_id * (self.window_size - self.overlap_size)
            end_t = start_t + data_sample.slice_len
            assert start_t < end_t
        elif self.slice_strategy == "random":
            seq_id = idx % self.num_sequences
            data_sample = self.data[seq_id]

            T = data_sample.sequence_len
            if data_sample.is_padded:
                start_t = 0
            else:
                start_t = np.random.randint(0, T - self.window_size + 1)
            end_t = start_t + min(self.window_size, T)
        else:
            raise ValueError(f"Unknown slice strategy {self.slice_strategy}")

        seq_name = data_sample.sequence
        gender = (
            self.h5datasets[seq_id]["attrs"]["gender"]
            if self.preload_files
            else self.h5datasets[seq_id].attrs["gender"]
        )
        subseq_len = end_t - start_t
        gender = "neutral"

        sbj_gender = 0 if gender == "neutral" else 1 if gender == "female" else 2

        # Initialize batch_dict
        batch_dict = {}
        for key in [
            "betas",
            "global_orient",
            "body_pose",
            "transl",
            "timestamps",
            "T_world_head",
            "body_pose_6d",
        ]:
            batch_dict[key] = self._get_tensor(seq_id, key, start_t, end_t)
        batch_dict["betas"] = batch_dict["betas"][:, : self.num_betas]
        batch_dict["T_world_head"] = batch_dict["T_world_head"].reshape(-1, 7)

        # Prepare data sample:
        #   1. Canonicalization of window
        #   2. Human pose
        #   3. Object pose
        #   4. Contacts
        # (1 / 4) ==> Obtain canonicalization
        # Every frame's rotation around Z is reset
        T_world_head = SE3(batch_dict["T_world_head"].clone())
        R_reset_z = get_reset_Z_rotation(T_world_head.rotation().as_matrix())
        T_can_world = SE3.from_rotation_and_translation(
            SO3.from_matrix(R_reset_z), torch.zeros(subseq_len, 3)
        )

        # Apply canonicalization to head and joints
        T_world_head = SE3(batch_dict["T_world_head"].clone())
        T_can_head = T_can_world @ T_world_head
        batch_dict["T_can_world"] = T_can_world.parameters()
        batch_dict["T_can_head"] = T_can_head.parameters()
        batch_dict["sbj_joints"] = self._get_tensor(
            seq_id, "smpl_joints_world", start_t, end_t
        )
        # <===

        # (2 / 4) ==> Obtain HMD features
        # Realtive head transformation
        T_head_tm1_head_t = SE3(
            self._get_tensor(seq_id, "T_head_tm1_head_t", start_t, end_t)
        ).as_matrix()
        R_head_tm1_head_t = matrix_to_rotation_6d(T_head_tm1_head_t[..., :3, :3])
        t_head_tm1_head_t = T_head_tm1_head_t[..., :3, 3]

        # Canonicalized head rotation
        T_world_head = SE3(batch_dict["T_world_head"].clone())
        R_can_head = (T_can_world @ T_world_head).rotation().as_matrix()
        R_can_head = matrix_to_rotation_6d(R_can_head)

        # Head height - Z component of translation
        head_height = T_world_head.translation()[..., [2]]

        # Realtive hands transformation
        T_hands_tm1_hands_t = SE3(
            self._get_tensor(seq_id, "T_hands_tm1_hands_t", start_t, end_t)
        ).as_matrix()
        R_hands_tm1_hands_t = matrix_to_rotation_6d(
            T_hands_tm1_hands_t[..., :3, :3]
        )
        t_hands_tm1_hands_t = T_hands_tm1_hands_t[..., :3, 3]

        # Canonicalized hands position
        T_can_world_unsqueezed = SE3(T_can_world.parameters().unsqueeze(1))
        T_world_hands = SE3(
            self._get_tensor(seq_id, "T_world_hands", start_t, end_t)
        )
        T_can_hands = (T_can_world_unsqueezed @ T_world_hands).as_matrix()
        R_can_hands = matrix_to_rotation_6d(T_can_hands[..., :3, :3])
        t_can_hands = T_can_hands[..., :3, 3]

        # Pre assemble hands for convenient nosie simulation
        cond_hands = torch.cat(
            [
                R_hands_tm1_hands_t.reshape(-1, 12),
                t_hands_tm1_hands_t.reshape(-1, 6),
                R_can_hands.reshape(-1, 12),
                t_can_hands.reshape(-1, 6),
            ],
            dim=-1,
        )

        # Simulate missing hand observation
        # We overwrite random portion of data with the last observed hand data
        if self.noise_simulation_percent > 0:
            cond_hands = simulate_observation_noise(
                cond_hands, self.noise_simulation_percent
            )

        hmd_condition = [
            R_head_tm1_head_t.reshape(-1, 6),
            t_head_tm1_head_t.reshape(-1, 3),
            head_height.reshape(-1, 1),
            R_can_head.reshape(-1, 6),
            cond_hands.reshape(-1, 36),
        ]
        batch_dict["hmd_condition"] = torch.cat(hmd_condition, dim=-1)

        # Prepare human pose for supervision
        batch_dict["body_pose_6d"] = batch_dict["body_pose_6d"].view(subseq_len, -1)
        # <===

        # (3 / 4) ==> Prepare object parameters
        object_metadata = {
            "obj_class_name": None,
            "obj_class_id": None,
            "obj_pointnext": None,
        }
        if self.include_object:
            # Extract global pose
            obj_R = self._get_tensor(seq_id, "obj_R", start_t, end_t)
            obj_transl = self._get_tensor(seq_id, "obj_transl", start_t, end_t)
            batch_dict["T_world_obj"] = (
                SE3.from_rotation_and_translation(
                    SO3.from_matrix(obj_R),
                    obj_transl,
                )
                .parameters()
                .float()
            )

            # Extract class information
            obj_class_name = (
                self.h5datasets[seq_id]["attrs"]["object_name"]
                if self.preload_files
                else self.h5datasets[seq_id].attrs["object_name"]
            )
            obj_class_id = self.obj2classid[obj_class_name]

            # Add pointnext conditioning
            if self.conditioning_config.use_pointnext_conditioning:
                object_metadata["obj_pointnext"] = self.obj_pointnext[
                    obj_class_id
                ].clone()
            else:
                object_metadata["obj_pointnext"] = None

            # Convert pose to relative if needed
            if self.model_config.object_pose == "canonical":
                T_can_obj = SE3(batch_dict["T_can_world"].clone()) @ SE3(
                    batch_dict["T_world_obj"].clone()
                )

                R_can_obj = T_can_obj.rotation().as_matrix()
                t_can_obj = T_can_obj.translation()

                obj_R_6d = matrix_to_rotation_6d(R_can_obj)
                obj_t_6d = t_can_obj
            elif self.model_config.object_pose == "relative":
                T_head_obj = SE3(batch_dict["T_world_head"].clone()).inverse() @ SE3(
                    batch_dict["T_world_obj"].clone()
                )
                R_head_obj = T_head_obj.rotation().as_matrix()
                t_head_obj = T_head_obj.translation()

                obj_R_6d = matrix_to_rotation_6d(R_head_obj)
                obj_t_6d = t_head_obj
            else:
                raise ValueError(f"Unknown object pose type {self.object_pose}")

            batch_dict["obj_Rt_6d"] = torch.cat([obj_R_6d, obj_t_6d], dim=-1).float()
            object_metadata["obj_class_id"] = obj_class_id
            object_metadata["obj_class_name"] = obj_class_name
        elif self.include_fake_object:
            R = torch.eye(3, dtype=torch.float)
            R_6d = matrix_to_rotation_6d(R)
            t = torch.zeros(3, dtype=torch.float)

            batch_dict["obj_Rt_6d"] = torch.cat(
                [R_6d.reshape(1, 6), t.reshape(1, 3)], dim=1
            ).repeat(subseq_len, 1)
            batch_dict["T_world_obj"] = (
                SE3.from_rotation_and_translation(SO3.from_matrix(R), t)
                .parameters()
                .reshape(1, 7)
                .repeat(subseq_len, 1)
            )

            if self._separate_fake_object:
                obj_class_name = "fake object"
                obj_class_id = max(list(self.obj2classid.values())) + 1
            else:
                obj_class_name = random.choice(list(self.obj2classid.keys()))
                obj_class_id = self.obj2classid[obj_class_name]

            object_metadata["obj_pointnext"] = torch.zeros_like(self.obj_pointnext[0])
            object_metadata["obj_class_id"] = obj_class_id
            object_metadata["obj_class_name"] = obj_class_name
        else:
            pass

        # network operates on the normalized object representation
        if "obj_Rt_6d" in batch_dict:
            batch_dict["obj_Rt_6d"] = normalize_obj_Rt_6d(batch_dict["obj_Rt_6d"])
        # <===

        # (4 / 4) ==> Prepare contacts
        if (
            self.model_config.include_env_contact
            or self.model_config.include_hoi_contact
        ):
            # Environment
            if self.model_config.include_env_contact:
                env_contacts = self._get_tensor(seq_id, "env_contacts", start_t, end_t)
                env_contacts = compute_env_contactness(env_contacts)
            else:
                env_contacts = torch.empty((subseq_len, 0), dtype=torch.float)

            # HOI
            if self.model_config.include_hoi_contact:
                if self.include_object and not self.include_fake_object:
                    hoi_contacts = self._get_tensor(
                        seq_id, "hoi_contacts", start_t, end_t
                    )
                    hoi_contacts = compute_hoi_contactness(hoi_contacts)
                else:
                    hoi_contacts = torch.zeros(
                        (subseq_len, self._num_downsampled_verts), dtype=torch.float
                    )
            else:
                hoi_contacts = torch.empty((subseq_len, 0), dtype=torch.float)

            batch_dict["contacts"] = torch.cat([hoi_contacts, env_contacts], dim=-1)
        else:
            batch_dict["contacts"] = None
        # <===

        # pad if needed
        if data_sample.is_padded:
            for k, v in batch_dict.items():
                if v is not None and v.shape[0] == data_sample.slice_len:
                    batch_dict[k] = torch.cat(
                        [
                            v,
                            torch.repeat_interleave(
                                v[-1:], self.window_size - data_sample.slice_len, dim=0
                            ),
                        ]
                    )

        return BatchData(
            # Object information
            is_fake_object=torch.tensor(self.include_fake_object, dtype=torch.bool),
            include_object=torch.tensor(self.include_object, dtype=torch.bool),
            # Sub-sequence information
            t_start=start_t,
            t_end=end_t,
            seq_name=seq_name,
            seq_id=seq_id,
            # Additional information
            sbj_gender=sbj_gender,
            real_slice_len=data_sample.slice_len,
            total_sequence_len=data_sample.sequence_len,
            is_end_of_sequence=data_sample.is_last,
            # Object metadata
            **object_metadata,
            # All params
            **batch_dict,
        )
