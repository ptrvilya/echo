from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
import smplx
import torch
import trimesh
from smplx.lbs import batch_rigid_transform, batch_rodrigues
from torch import Tensor, nn

from echo.configs.environment import EnvironmentConfig
from echo.data.batch_data import BatchData
from echo.model.output import EgoModelOutput
from echo.utils.rotation import aa_to_matrix, matrix_to_aa, rotation_6d_to_matrix
from echo.utils.smpl import local2global_pose
from echo.utils.transforms import SE3, SO3


@dataclass
class MeshOutput:
    sbj_vertices: Optional[Union[torch.Tensor, npt.NDArray]]
    sbj_joints: Optional[Union[torch.Tensor, npt.NDArray]]
    obj_vertices: Optional[Union[torch.Tensor, npt.NDArray]]
    obj_keypoints: Optional[Union[torch.Tensor, npt.NDArray]]


class FastJointSMPLX(nn.Module):
    """
    A lightweight wrapper around SMPLX that skips LBS and vertex blendshapes.
    Calculates only 3D joint positions for massive speed/memory improvements.
    """

    def __init__(self, smplx_model: smplx.SMPLX):
        super().__init__()
        self.dtype = smplx_model.v_template.dtype
        self.register_buffer("parents", smplx_model.parents.long().detach().clone())

        # -----------------------------------------------------------
        # 1. Precompute Rest Joints (J_template)
        # J_regressor [J, V] x Template Vertices [V, 3] -> [J, 3]
        # -----------------------------------------------------------
        J_template = torch.matmul(
            smplx_model.J_regressor.detach().clone(),
            smplx_model.v_template.detach().clone(),
        )
        self.register_buffer("J_template", J_template)

        # -----------------------------------------------------------
        # 2. Precompute Joint Shape Directions (J_shapedirs)
        # Project vertex shape blendshapes directly into joint space
        # -----------------------------------------------------------
        num_betas = smplx_model.shapedirs.shape[-1]
        num_vertices = smplx_model.v_template.shape[0]

        # Reshape [V, 3, num_betas] -> [V, 3 * num_betas]
        shapedirs_flat = (
            smplx_model.shapedirs.detach().clone().reshape(num_vertices, -1)
        )

        # Multiply [J, V] x [V, 3 * num_betas] -> [J, 3 * num_betas]
        J_shapedirs = torch.matmul(
            smplx_model.J_regressor.detach().clone(), shapedirs_flat
        )

        # Reshape back to [J, 3, num_betas]
        J_shapedirs = J_shapedirs.reshape(-1, 3, num_betas)
        self.register_buffer("J_shapedirs", J_shapedirs)

    def forward(self, betas, global_orient, body_pose, transl=None, **kwargs):
        """
        Inputs expect standard SMPL-X axis-angle formats.
        betas: [B, num_betas]
        global_orient: [B, 3]
        body_pose: [B, J*3] (e.g., 63 for SMPL-X's 21 body joints)
        transl: [B, 3] (Optional)
        """
        batch_size = betas.shape[0]
        device = betas.device

        # 1. Shape to Rest Joints (Ultra-fast, skips 10k vertices)
        # Einsum: j=joints, d=dimensions(3), b=betas, n=batch
        J_shape_disp = torch.einsum("jdb,nb->njd", self.J_shapedirs, betas)
        J_rest = self.J_template.unsqueeze(0) + J_shape_disp

        # 2. Handle missing hand/face joints
        # SMPL-X expects 55 total joints (1 global + 21 body + hands + face)
        # We pad missing joints with zeros (identity rotation in axis-angle)
        num_total_joints = len(self.parents)
        num_provided_joints = 1 + (body_pose.shape[1] // 3)
        num_missing_joints = num_total_joints - num_provided_joints

        if num_missing_joints > 0:
            padding = torch.zeros(
                batch_size, num_missing_joints * 3, device=device, dtype=self.dtype
            )
            full_pose_aa = torch.cat([global_orient, body_pose, padding], dim=1)
        else:
            full_pose_aa = torch.cat([global_orient, body_pose], dim=1)

        # 3. Convert Axis-Angle to Rotation Matrices
        full_pose_flat = full_pose_aa.reshape(-1, 3)
        rot_mats = batch_rodrigues(full_pose_flat)
        rot_mats = rot_mats.view(batch_size, -1, 3, 3)

        # 4. Forward Kinematics (using exact smplx method)
        # Returns J_posed and A (relative transforms for LBS, which we discard)
        J_posed, _ = batch_rigid_transform(
            rot_mats, J_rest, self.parents, dtype=self.dtype
        )

        # 5. Apply Global Translation
        if transl is not None:
            J_posed = J_posed + transl.unsqueeze(1)

        return {"joints": J_posed}


class MeshModel:
    """
    A class that genrate human and object mesh given the parameters
    predicted by the denoising model.
    """

    def __init__(
        self,
        batch_size: int,
        env: EnvironmentConfig,
        device="cpu",
        num_betas: Optional[int] = 10,
        model_type="smplh",
        include_object=False,
        canonical_obj_meshes: Dict[int, trimesh.Trimesh] = None,
        canonical_obj_keypoints: Dict[int, Dict[str, npt.NDArray]] = None,
        inference_chunk_size: int = 32,
        infer_joints_only: bool = False,
    ) -> None:
        self.num_betas = num_betas
        self.model_type = model_type
        if model_type == "smplh":
            self.smpl = smplx.SMPLH(
                model_path=env.smplh_amass_folder,
                model_type="smplh",
                gender="male",
                use_pca=False,
                num_betas=self.num_betas,
                batch_size=1,
                flat_hand_mean=True,
            ).to(device)

            self.smpl_downsample_indices = torch.arange(0, 6890, device=device)
        elif model_type == "smplx":
            self.smpl = smplx.SMPLX(
                model_path=env.smplx_folder,
                model_type="smplx",
                gender="neutral",
                use_pca=False,
                num_betas=self.num_betas,
                batch_size=1,
                flat_hand_mean=True,
            ).to(device)
            self.smpl_downsample_indices = torch.tensor(
                np.load(Path(env.assets_folder) / "smplx_downsample_indices.npy"),
                device=device,
            )

        # Used in dataset to recalculate features
        self.smpl_parents = self.smpl.parents.detach().cpu().long()
        self._head_id: int = 15  # head joint id in SMPL model

        # Swap for a optimised joints-only model
        if infer_joints_only:
            self.smpl = FastJointSMPLX(self.smpl).to(device)

        # batch size
        self.batch_size = batch_size
        self.inference_chunk_size = inference_chunk_size
        self.device = device

        # object
        self.include_object = include_object
        self.canonical_obj_meshes = canonical_obj_meshes
        self.canonical_obj_keypoints = canonical_obj_keypoints

        if include_object:
            class_ids = sorted(self.canonical_obj_keypoints.keys())
            self.canonical_obj_keypoints_th = [
                self.canonical_obj_keypoints[class_id]["cartesian"]
                for class_id in class_ids
            ]
            # extra all-zero row for the fake object class (id = max real id + 1)
            self.canonical_obj_keypoints_th.append(
                np.zeros_like(self.canonical_obj_keypoints_th[0])
            )
            self.canonical_obj_keypoints_th = torch.tensor(
                np.stack(self.canonical_obj_keypoints_th, axis=0),
                dtype=torch.float,
                requires_grad=False,
                device=device,
            )

            # Per-class canonical mesh vertices/faces (variable vertex counts,
            # so kept as dicts rather than a padded tensor).
            self.canonical_obj_verts_th: Dict[int, torch.Tensor] = {}
            self.canonical_obj_faces_th: Dict[int, torch.Tensor] = {}
            if canonical_obj_meshes is not None:
                for class_id, mesh in canonical_obj_meshes.items():
                    self.canonical_obj_verts_th[class_id] = torch.tensor(
                        np.asarray(mesh.vertices, dtype=np.float32),
                        dtype=torch.float,
                        device=device,
                    )
                    self.canonical_obj_faces_th[class_id] = torch.tensor(
                        np.asarray(mesh.faces, dtype=np.int64),
                        dtype=torch.long,
                        device=device,
                    )

    def local_to_global(
        self,
        params: Dict[str, Tensor],
        vertices: Tensor,
        joints: Tensor,
        T_world_head: Tensor,
        with_skin: bool = False,
        inference_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, T = joints.shape[:2]
        device = self.device

        for k in params.keys():
            params[k] = params[k].to(device)
        joints = joints.to(device)
        T_world_head = T_world_head.to(device)
        vertices = vertices.to(device)

        if inference_mask is not None:
            mask = inference_mask.view(B, T)
        else:
            mask = torch.ones((B, T), dtype=torch.bool, device=device)

        if not mask.any():
            return vertices, joints

        # Convert to global rotmats (in world coordinates)
        # 1 / 3 construct tensor with local aa pose
        pose_aa_local = [params["global_orient"], params["body_pose"]]
        if self.model_type == "smplx":
            pose_aa_local.extend(
                [
                    torch.zeros((B, T, 3), dtype=torch.float, device=device),
                    torch.zeros((B, T, 3), dtype=torch.float, device=device),
                    torch.zeros((B, T, 3), dtype=torch.float, device=device),
                ]
            )
        pose_aa_local.extend([params["left_hand_pose"], params["right_hand_pose"]])
        pose_aa_local = torch.cat(pose_aa_local, dim=2)

        # Mask
        pose_aa_local = pose_aa_local[mask]
        N_MASKED = pose_aa_local.shape[0]

        # 2 / 3 convert local pose from aa to matrix
        pose_aa_local = pose_aa_local.reshape(-1, 3)  # N_MASKED * N_J, 3
        pose_matrix_local = aa_to_matrix(pose_aa_local)
        # 3 / 3 convert local pose to global, following kinematic tree
        pose_matrix_global = local2global_pose(
            pose_matrix_local.reshape(N_MASKED, -1, 3, 3), self.smpl_parents.to(device)
        )  # N_MASKED, N_J, 3, 3

        # Obtain correct root transformation by aligning head rotation and translation
        T_world_head = SE3(T_world_head.reshape(B, T, -1)[mask])
        joints_masked = joints[mask]

        old_T_world_head = SE3.from_rotation_and_translation(
            SO3.from_matrix(pose_matrix_global[:, self._head_id]),
            joints_masked[:, self._head_id],
        )
        T_world_root = SE3.from_rotation_and_translation(
            SO3.from_matrix(pose_matrix_global[:, 0]), joints_masked[:, 0]
        )

        T_world_head = T_world_head @ old_T_world_head.inverse() @ T_world_root
        T_world_head = T_world_head.as_matrix()
        R_world_head = T_world_head[:, :3, :3]
        t_world_head = (
            T_world_head[:, :3, 3] - joints_masked[:, 0]
        )  # correct for root offset

        # update params for correct inference if needed
        params["global_orient"][mask] = matrix_to_aa(R_world_head)
        params["transl"][mask] += t_world_head

        # update vertices and joints without second inference
        if with_skin:
            vertices_masked = vertices[mask]
            vertices_masked = (
                torch.matmul(
                    R_world_head,
                    (vertices_masked - joints_masked[:, [0]]).transpose(2, 1),
                ).transpose(2, 1)
                + t_world_head.unsqueeze(1)
                + joints_masked[:, [0]]
            )
            vertices[mask] = vertices_masked

        joints_masked = (
            torch.einsum(
                "nij,nkj->nki", R_world_head, (joints_masked - joints_masked[:, [0]])
            )
            + t_world_head.unsqueeze(1)
            + joints_masked[:, [0]]
        )
        joints[mask] = joints_masked

        return vertices, joints

    def get_meshes_from_batch(
        self,
        batch: BatchData,
        inference_mask=None,
        with_skin: bool = False,
    ) -> MeshOutput:
        B, T, _ = batch.betas.shape

        body_model_params = {
            "betas": batch.betas.reshape(B, T, self.num_betas),
            "transl": batch.transl.reshape(B, T, 3),
            "global_orient": batch.global_orient.reshape(B, T, 3),
            "body_pose": batch.body_pose.reshape(B, T, 21 * 3),
            "left_hand_pose": torch.zeros(B, T, 15 * 3),
            "right_hand_pose": torch.zeros(B, T, 15 * 3),
        }

        # get sbj mesh
        sbj_vertices, sbj_joints = self.get_smpl_th(
            body_model_params,
            with_skin=with_skin,
            inference_mask=inference_mask,
        )

        # pose the objects' keypoints
        if self.include_object:
            # B x N x 3 -> B x 1 x N x 3 -> B x T x N x 3
            object_keypoints = self.canonical_obj_keypoints_th[
                batch.obj_class_id
            ].clone()
            object_keypoints = object_keypoints.unsqueeze(1)
            object_keypoints = object_keypoints.repeat_interleave(T, 1).to(
                batch.T_world_obj.device
            )

            T_world_obj = SE3(batch.T_world_obj)
            obj_R = T_world_obj.rotation().as_matrix().reshape(B, T, 3, 3)
            obj_t = T_world_obj.translation().unsqueeze(2)
            object_keypoints = torch.matmul(
                obj_R, object_keypoints.transpose(3, 2)
            ).transpose(3, 2)
            object_keypoints += obj_t
        else:
            object_keypoints = None

        return MeshOutput(
            sbj_vertices=sbj_vertices,
            sbj_joints=sbj_joints,
            obj_vertices=None,
            obj_keypoints=object_keypoints,
        )

    def get_meshes_from_output(
        self,
        output: EgoModelOutput,
        return_joints: bool = False,
        local_to_global: bool = False,
        inference_mask=None,
        with_skin: bool = False,
    ) -> MeshOutput:
        B, T, _ = output.body_pose_6d.shape

        body_pose_6d = output.body_pose_6d.reshape(B * T, -1, 6)
        body_pose_aa = matrix_to_aa(rotation_6d_to_matrix(body_pose_6d)).reshape(
            B, T, 21 * 3
        )

        # Default global pose
        global_pose_aa = torch.zeros(
            (B, T, 3), device=body_pose_6d.device, dtype=torch.float
        )
        # move zeros to output?
        body_model_params = {
            "betas": output.betas.reshape(B, T, self.num_betas),
            "transl": output.transl.reshape(B, T, 3)
            if output.transl is not None
            else torch.zeros(B, T, 3),
            "global_orient": global_pose_aa,
            "body_pose": body_pose_aa,
            "left_hand_pose": torch.zeros(B, T, 15 * 3),
            "right_hand_pose": torch.zeros(B, T, 15 * 3),
        }

        # get sbj mesh
        sbj_vertices, sbj_joints = self.get_smpl_th(
            body_model_params,
            with_skin=with_skin,
            inference_mask=inference_mask,
        )

        # convert predictions to the correct global frame
        if local_to_global and output is not None:
            assert output.T_world_head is not None, (
                "T_world_head is required for local_to_global"
            )
            sbj_vertices, sbj_joints = self.local_to_global(
                body_model_params,
                sbj_vertices,
                sbj_joints,
                output.T_world_head,
                with_skin=with_skin,
                inference_mask=inference_mask,
            )
            output.transl = body_model_params["transl"]
            output.global_orient = body_model_params["global_orient"]

        # pose the objects' keypoints
        if self.include_object:
            # B x N x 3 -> B x 1 x N x 3 -> B x T x N x 3
            object_keypoints = self.canonical_obj_keypoints_th[
                output.obj_class_id
            ].clone()
            object_keypoints = object_keypoints.unsqueeze(1)
            object_keypoints = object_keypoints.repeat_interleave(T, 1)

            T_world_obj = SE3(output.T_world_obj)
            obj_R = T_world_obj.rotation().as_matrix().reshape(B, T, 3, 3)
            obj_t = T_world_obj.translation().unsqueeze(2)
            object_keypoints = torch.matmul(
                obj_R, object_keypoints.transpose(3, 2)
            ).transpose(3, 2)
            object_keypoints += obj_t
        else:
            object_keypoints = None

        return MeshOutput(
            sbj_vertices=sbj_vertices,
            sbj_joints=sbj_joints if return_joints else None,
            obj_vertices=None,
            obj_keypoints=object_keypoints,
        )

    def pose_object_vertices(
        self, object_class_id: int, obj_R: torch.Tensor, obj_t: torch.Tensor
    ) -> torch.Tensor:
        """Pose canonical object vertices for a single class with per-frame R, t.

        obj_R: (T, 3, 3), obj_t: (T, 3) -> returns (T, V, 3).
        """
        if not self.include_object or not self.canonical_obj_verts_th:
            raise ValueError(
                "MeshModel must be initialized with canonical_obj_meshes "
                "to pose object vertices."
            )
        v_can = self.canonical_obj_verts_th[object_class_id]
        obj_R = obj_R.to(v_can.device)
        obj_t = obj_t.to(v_can.device)
        # (T, 3, 3) @ (V, 3).T -> (T, 3, V) -> (T, V, 3)
        v_posed = torch.matmul(obj_R, v_can.T.unsqueeze(0)).transpose(1, 2)
        v_posed = v_posed + obj_t.unsqueeze(1)
        return v_posed

    def get_object_faces(self, object_class_id: int) -> torch.Tensor:
        return self.canonical_obj_faces_th[object_class_id]

    def get_meshes_from_h5(
        self,
        h5_file: h5py.File,
        object_class_id: int = None,
        with_skin: bool = False,
        with_object_verts: bool = False,
    ) -> MeshOutput:
        f = h5_file
        T = f.attrs["T"]
        with torch.no_grad():
            device = self.device

            # Load SMPL params
            betas = torch.from_numpy(f["betas"][:]).to(device).float()
            if betas.shape[0] != T:
                betas = betas.reshape(1, -1).expand(T, -1)
            betas = betas[:, : self.num_betas]

            global_orient = torch.from_numpy(f["global_orient"][:]).to(device).float()
            transl = torch.from_numpy(f["transl"][:]).to(device).float()
            body_pose = torch.from_numpy(f["body_pose"][:]).to(device).float()

            # Create BatchData with B=1
            batch = BatchData(
                betas=betas.unsqueeze(0),
                global_orient=global_orient.unsqueeze(0),
                body_pose=body_pose.unsqueeze(0),
                transl=transl.unsqueeze(0),
            )

            if self.include_object and object_class_id is not None:
                obj_R = torch.from_numpy(f["obj_R"][:]).to(device).float()
                if "obj_transl" in f:
                    obj_t = torch.from_numpy(f["obj_transl"][:]).to(device).float()
                else:
                    obj_t = torch.from_numpy(f["obj_Rt_6d"][:, 6:]).to(device).float()

                # Construct SE3 parameters
                T_world_obj = (
                    SE3.from_rotation_and_translation(SO3.from_matrix(obj_R), obj_t)
                    .parameters()
                    .unsqueeze(0)
                )
                batch.T_world_obj = T_world_obj
                batch.obj_class_id = torch.tensor([object_class_id], device=device)

        mesh_output = self.get_meshes_from_batch(batch, with_skin=with_skin)

        if with_object_verts and self.include_object and object_class_id is not None:
            obj_vertices = self.pose_object_vertices(object_class_id, obj_R, obj_t)
            mesh_output.obj_vertices = obj_vertices.unsqueeze(0)

        return mesh_output

    def get_smpl_th(
        self,
        params: Dict,
        with_skin: bool = False,
        inference_mask=None,
    ):
        B, T, _ = params["betas"].shape

        if self.model_type == "smplx":
            params.update(
                {
                    "expression": torch.zeros((B, T, 10), dtype=torch.float),
                    "jaw_pose": torch.zeros((B, T, 3), dtype=torch.float),
                    "leye_pose": torch.zeros((B, T, 3), dtype=torch.float),
                    "reye_pose": torch.zeros((B, T, 3), dtype=torch.float),
                }
            )

        params = {
            k: v.reshape(B * T, -1).to(self.device) if v is not None else None
            for k, v in params.items()
        }

        # Infer in chunks
        if with_skin:
            vertices = torch.zeros(
                (B * T, self.smpl.v_template.shape[0], 3),
                dtype=torch.float,
                device=self.device,
            )
        else:
            vertices = torch.empty(0, device=self.device)
        joints = torch.zeros((B * T, 55, 3), dtype=torch.float, device=self.device)

        # mask
        if inference_mask is None:
            inference_mask = torch.ones((B * T,), dtype=torch.bool, device=self.device)
        else:
            inference_mask = inference_mask.reshape(B * T)

        params_masked = {k: v[inference_mask] for k, v in params.items()}
        N_MASKED: int = params_masked["betas"].shape[0]
        joints_masked = joints[inference_mask]
        if with_skin:
            vertices_masked = vertices[inference_mask]

        for chunk in range(0, N_MASKED, self.inference_chunk_size):
            chunk_start = chunk
            chunk_end = chunk + self.inference_chunk_size
            params_chunk = {
                k: v[chunk_start:chunk_end] for k, v in params_masked.items()
            }

            # get smpl vertices
            smpl_output = self.smpl(**params_chunk)

            joints_chunk = smpl_output["joints"][:, :55]
            joints_masked[chunk_start:chunk_end] = joints_chunk

            if with_skin:
                vertices_chunk = smpl_output["vertices"]
                vertices_masked[chunk_start:chunk_end] = vertices_chunk

        joints[inference_mask] = joints_masked
        joints = joints.reshape(B, T, -1, 3)
        if with_skin:
            vertices[inference_mask] = vertices_masked
            vertices = vertices.reshape(B, T, -1, 3)

        return vertices, joints
