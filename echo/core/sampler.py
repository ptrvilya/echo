from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from echo.configs import ProjectConfig
from echo.data import get_eval_dataloaders
from echo.data.batch_data import BatchData
from echo.data_parsers.common import export_sequence_to_hdf5
from echo.model.base import BaseEgoModel
from echo.model.inference import get_inference_model
from echo.model.mesh_model import MeshModel
from echo.model.output import EgoModelOutput
from echo.utils.rotation import (
    matrix_to_aa,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from echo.utils.training import resume_from_checkpoint
from echo.utils.transforms import SE3


class Sampler:
    def __init__(self, config: ProjectConfig, model: BaseEgoModel):
        self.logger = getLogger(__name__)

        self.config = config
        self.model = model.cuda()
        self.device = torch.device("cuda")

        # Resume from checkpoint and create the initial training state
        self.train_state, _ = resume_from_checkpoint(config, model, None, None)

        # Get dataloaders
        # turn off preloading for datasets that are not used
        self.dataloaders, obj_keypoints = get_eval_dataloaders(config)

        # Create mesh model to obtain posed joints
        self.mesh_model = MeshModel(
            model_type=config.model_mesh.type,
            env=config.env,
            num_betas=config.model_mesh.num_betas,
            include_object=config.model_mesh.include_object or config.model.cg_apply,
            device=self.device,
            inference_chunk_size=config.model_mesh.inference_chunk_size,
            batch_size=config.model_mesh.inference_chunk_size,
            canonical_obj_keypoints=obj_keypoints,
            infer_joints_only=False,
        )
        self.model.set_mesh_model(self.mesh_model)

        # Create inference model for validation
        self.inference_model = get_inference_model(config)

        # Create folder for artifacts
        base_name = "artifacts"
        if len(self.config.sample.artifacts_folder_suffix) > 0:
            base_name = f"{base_name}_{self.config.sample.artifacts_folder_suffix}"

        self.base_samples_folder = (
            Path(self.config.run.path)
            / base_name
            / f"step_{self.config.resume.step}_samples"
        )
        self.base_samples_folder.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def sample_step(self, batch):
        self.model.eval()

        sample_mode = self.config.sample.modalities

        output = self.model(
            batch.to("cuda"),
            self.config.sample.mode,
            sample_type=sample_mode,
            scheduler=self.config.sample.scheduler,
            num_inference_steps=self.config.sample.num_inference_steps,
            inference_model=self.inference_model,
        )

        return output

    @staticmethod
    def sample_mode_to_str(sample_mode, dataset_name):
        sample_str = []
        if sample_mode[0] == "1":
            sample_str.append("sbj")
        if (
            len(sample_mode) > 1
            and sample_mode[1] == "1"
            and dataset_name in ["omomo", "behave"]
        ):
            sample_str.append("obj")
        if (
            len(sample_mode) > 2
            and sample_mode[2] == "0"
            and dataset_name in ["omomo", "behave"]
        ):
            sample_str.append("from_contact")

        sample_str = "_".join(sample_str)

        return sample_str

    @staticmethod
    def _init_sample_file(
        samples_folder: Path,
        sequence_name: str,
        output,
        is_fake_object: bool = False,
        metadata: dict[Any, Any] | None = None,
    ):
        # Output file
        hdf5_path = samples_folder / f"{sequence_name}.hdf5"

        # Padded sequence length
        T_padded = output.body_pose_6d.shape[1]
        body_pose_6d = output.body_pose_6d.reshape(T_padded, -1, 6)
        body_pose_aa = matrix_to_aa(rotation_6d_to_matrix(body_pose_6d))

        # Body model params
        bm_params = {
            "betas": output.betas,
            "global_orient": output.global_orient,
            "body_pose": body_pose_aa,
            "transl": output.transl,
        }

        # Motion features
        motion_features = {
            "T_world_head": output.T_world_head,
            "smpl_joints_world": output.sbj_joints,
        }

        # Object
        object_features = {}
        if output.T_world_obj is not None and not (is_fake_object):
            # assumes batch_size == 1
            T_world_obj = SE3(output.T_world_obj).as_matrix().squeeze(0).cpu()

            obj_global_R = T_world_obj[:, :3, :3]
            obj_transl = T_world_obj[:, :3, 3]
            obj_global_Rt_6d = torch.cat(
                [matrix_to_rotation_6d(obj_global_R), obj_transl], dim=-1
            )

            object_features["obj_Rt_6d"] = obj_global_Rt_6d.numpy()
            object_features["obj_R"] = obj_global_R.numpy()
            object_features["obj_transl"] = obj_transl.numpy()

        # Contacts
        if output.diff_contacts is not None:
            object_features["diff_contacts"] = output.diff_contacts.cpu().numpy()

        # Trim to real sequence length
        T = output.real_slice_len.item()

        def _maybe_squeeze_first_dim(x):
            return x.squeeze(0) if x.size(0) == 1 else x

        def _trim_and_convert(x):
            if isinstance(x, torch.Tensor):
                x = _maybe_squeeze_first_dim(x)
                x = x[:T]
                return x.cpu().numpy()
            else:
                return x

        bm_params = {k: _trim_and_convert(v) for k, v in bm_params.items()}
        motion_features = {k: _trim_and_convert(v) for k, v in motion_features.items()}
        object_features = {k: _trim_and_convert(v) for k, v in object_features.items()}

        # Metadata
        metadata["T"] = T

        export_sequence_to_hdf5(
            bm_params=bm_params,
            motion_features=motion_features,
            object_features=object_features,
            metadata=metadata,
            output_path=hdf5_path,
        )

    @torch.no_grad()
    def sample(self):
        # Log general info
        self.logger.info(
            f"***** Starting sampling *****\n"
            f"    Model: {self.config.model_denoising.name}\n"
            f"    Checkpoint step number: {self.config.resume.step}\n"
            f"    Repetitions: {self.config.sample.repetition_id}\n"
        )

        for dataloader in self.dataloaders:
            # Log info
            self.logger.info(
                f"    Sampling mode {self.config.sample.modalities} for: {dataloader.dataset.name}\n"
                f"    Number of samples: {len(dataloader.dataset)}\n"
            )

            sample_mode = self.sample_mode_to_str(
                self.config.sample.modalities, dataloader.dataset.name
            )
            sequence_id, batches, output = -1, [], EgoModelOutput()
            for repetition_id in self.config.sample.repetition_id:
                samples_folder = (
                    self.base_samples_folder
                    / f"{dataloader.dataset.name}_{dataloader.dataset.split}"
                    / f"{sample_mode}"
                    / f"repetition_{repetition_id}"
                )
                samples_folder.mkdir(parents=True, exist_ok=True)

                for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    ncols=80,
                    desc=f"Sampling {sample_mode} / Rep {repetition_id}",
                    total=len(dataloader),
                ):
                    if sequence_id != batch.seq_id[0].item():
                        sequence_id = batch.seq_id[0].item()
                        sequence_name = batch.seq_name[0]

                    # Get outputs
                    batches.append(batch)
                    _output = self.sample_step(batch)

                    # Exceptionally accumulate ourselves
                    if self.config.sample.mode == "sample_naive":
                        if output.is_empty():
                            output = _output
                        else:
                            for k, v in _output.items():
                                if v is not None:
                                    if k == "real_slice_len":
                                        output[k] += batch.real_slice_len[0].item()
                                    else:
                                        output[k] = torch.cat(
                                            [output[k], v],
                                            dim=1,
                                        )
                    else:
                        output = _output

                    # Save outputs to h5file
                    # Assume batch_size == 1
                    if batch.is_end_of_sequence[0]:
                        # Store all input batches for the sequence
                        sequence_batch = BatchData.concatenate(batches)

                        # Infer SMPL joints
                        mesh_output = self.mesh_model.get_meshes_from_output(
                            output,
                            local_to_global=True,
                            return_joints=True,
                        )
                        output.sbj_joints = mesh_output.sbj_joints

                        # Save result to hdf5
                        gender = sequence_batch.sbj_gender[0].item()
                        assert isinstance(gender, int)
                        gender_text = (
                            "neutral"
                            if gender == 0
                            else "female"
                            if gender == 1
                            else "male",
                        )
                        metadata = {
                            "gender": gender_text,
                            "framerate": dataloader.dataset.fps,
                            "sequence_name": sequence_name,
                        }
                        if sequence_batch.obj_class_name[0] is not None:
                            metadata["object_name"] = sequence_batch.obj_class_name[0][
                                0
                            ].encode("utf-8")

                        self._init_sample_file(
                            samples_folder,
                            sequence_name,
                            output,
                            metadata=metadata,
                        )

                        # Reset
                        output, batches = EgoModelOutput(), []
