"""
Provides functions for converting between different representations of rotations.
Uses quaternions as an intermediate representation.

Functions:
    SE3_to_Rt_6d: Convert SE3 transformation to 6D rotation + translation representation
    aa_to_quat: Convert axis-angle to quaternion
    quat_to_matrix: Convert quaternion to rotation matrix
    matrix_to_quat: Convert rotation matrix to quaternion
    quat_to_aa: Convert quaternion to axis-angle
    aa_to_matrix: Convert axis-angle to rotation matrix through quaternions
    matrix_to_aa: Convert rotation matrix to axis-angle through quaternions
    rotation_6d_to_matrix: Convert 6D rotation representation to rotation matrix
    matrix_to_rotation_6d: Convert rotation matrix to 6D rotation representation
    standardize_quaternion: Convert unit quaternion to standard form (non-negative real part)
"""

from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from typing_extensions import overload

from echo.utils.transforms import SE3


def SE3_to_Rt_6d(T: SE3) -> torch.Tensor:
    """
    Converts SE3 transformation to 6D rotation + translation representation.

    Args:
        T: SE3 transformation of shape (..., 4, 4)

    Returns:
        Rt_6d: 6D rotation + translation representation of shape (..., 9)
    """
    R = T.rotation().as_matrix()  # (..., 3, 3)
    t = T.translation()  # (..., 3)

    R_6d = matrix_to_rotation_6d(R)  # (..., 6)

    Rt_6d = torch.cat([R_6d, t], dim=-1)  # (..., 9)

    return Rt_6d


@overload
def aa_to_quat(aa: npt.NDArray, eps: float = 1e-6) -> npt.NDArray: ...


@overload
def aa_to_quat(aa: torch.Tensor, eps: float = 1e-6) -> torch.Tensor: ...


def aa_to_quat(
    aa: Union[npt.NDArray, torch.Tensor], eps: float = 1e-6
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Converts axis-angle representation to quaternion.

    Args:
        aa: axis-angle representation, of size (*, 3)
        eps: epsilon to determine small angles

    Returns:
        quaternion: batch of quaternions of size (*, 4) in (w, x, y, z) format
    """
    if isinstance(aa, np.ndarray):
        aa = torch.from_numpy(aa).float()
        is_numpy = True
    else:
        is_numpy = False

    if aa.dim() == 1:
        aa = aa.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    quat = torch.zeros(aa.size()[:-1] + (4,), device=aa.device)
    angles = aa.norm(p=2, dim=-1, keepdim=True)
    small_angles_mask = angles.abs() < eps
    half_angles = angles * 0.5
    sin_half_angles_over_angles = torch.empty_like(angles)

    # Handle large angles
    sin_half_angles_over_angles[~small_angles_mask] = (
        torch.sin(half_angles[~small_angles_mask]) / angles[~small_angles_mask]
    )

    # Handle small angles
    # Taylor approximation sin(x/2) ≈ x/2 - (x/2)³/6
    # Therefore: sin(x/2)/x ≈ 1/2 - x²/48
    sin_half_angles_over_angles[small_angles_mask] = (
        0.5 - (angles[small_angles_mask] * angles[small_angles_mask]) / 48
    )

    # Compute quaternion
    quat[..., :] = torch.concatenate(
        [torch.cos(half_angles), aa * sin_half_angles_over_angles], dim=-1
    )

    if is_squeeze:
        quat = quat.squeeze(0)

    if is_numpy:
        quat = quat.numpy()

    return quat


@overload
def quat_to_matrix(quat: npt.NDArray) -> npt.NDArray: ...


@overload
def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor: ...


def quat_to_matrix(
    quat: Union[npt.NDArray, torch.Tensor],
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Converts quaternion to rotation matrix.
    Quaternion is assumed to be in (w, x, y, z) format.

    Args:
        quat: quaternion, of size (*, 4) in (w, x, y, z) format

    Returns:
        matrix: batch of rotation matrices of size (*, 3, 3)

    """
    if isinstance(quat, np.ndarray):
        quat = torch.from_numpy(quat).float()
        is_numpy = True
    else:
        is_numpy = False

    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    w, x, y, z = torch.split(quat, 1, dim=-1)

    matrix = torch.stack(
        [
            torch.cat(
                [
                    1 - 2 * y * y - 2 * z * z,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x * x - 2 * z * z,
                    2 * y * z - 2 * x * w,
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x * x - 2 * y * y,
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )

    if is_squeeze:
        matrix = matrix.squeeze(0)

    if is_numpy:
        matrix = matrix.numpy()

    return matrix


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non-negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


@overload
def matrix_to_quat(matrix: npt.NDArray) -> npt.NDArray: ...


@overload
def matrix_to_quat(matrix: torch.Tensor) -> torch.Tensor: ...


def matrix_to_quat(
    matrix: Union[npt.NDArray, torch.Tensor],
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Convert rotations given as rotation matrices to quaternions using Shepperd's method.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
        is_numpy = True
    else:
        is_numpy = False

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    quat = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    quat = standardize_quaternion(quat)

    if is_squeeze:
        quat = quat.squeeze(0)

    if is_numpy:
        quat = quat.numpy()

    return quat


@overload
def quat_to_aa(quaternions: npt.NDArray) -> npt.NDArray: ...


@overload
def quat_to_aa(quaternions: torch.Tensor) -> torch.Tensor: ...


def quat_to_aa(
    quaternions: Union[npt.NDArray, torch.Tensor], eps: float = 1e-6
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    if isinstance(quaternions, np.ndarray):
        quaternions = torch.from_numpy(quaternions).float()
        is_numpy = True
    else:
        is_numpy = False

    if quaternions.dim() == 2:
        quaternions = quaternions.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    matrix = quaternions[..., 1:] / sin_half_angles_over_angles

    if is_squeeze:
        matrix = matrix.squeeze(0)

    if is_numpy:
        matrix = matrix.numpy()

    return matrix


@overload
def aa_to_matrix(aa: npt.NDArray) -> npt.NDArray: ...


@overload
def aa_to_matrix(aa: torch.Tensor) -> torch.Tensor: ...


def aa_to_matrix(
    aa: Union[npt.NDArray, torch.Tensor],
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Converts axis-angle representation to rotation matrix.

    Args:
        aa: axis-angle representation, of size (*, 3)

    Returns:
        matrix: batch of rotation matrices of size (*, 3, 3)
    """
    if isinstance(aa, np.ndarray):
        aa = torch.from_numpy(aa).float()
        is_numpy = True
    else:
        is_numpy = False

    if aa.dim() == 1:
        aa = aa.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    quat = aa_to_quat(aa)
    matrix = quat_to_matrix(quat)

    if is_squeeze:
        matrix = matrix.squeeze(0)

    if is_numpy:
        matrix = matrix.numpy()

    return matrix


@overload
def matrix_to_aa(matrix: npt.NDArray) -> npt.NDArray: ...


@overload
def matrix_to_aa(matrix: torch.Tensor) -> torch.Tensor: ...


def matrix_to_aa(
    matrix: Union[npt.NDArray, torch.Tensor],
) -> Union[npt.NDArray, torch.Tensor]:
    """
    Converts rotation matrix to axis-angle representation.

    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        aa: axis-angle representation, of size (*, 3)
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
        is_numpy = True
    else:
        is_numpy = False

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    quat = matrix_to_quat(matrix)
    aa = quat_to_aa(quat)

    if is_squeeze:
        aa = aa.squeeze(0)

    if is_numpy:
        aa = aa.numpy()
    return aa


@overload
def rotation_6d_to_matrix(d6: npt.NDArray) -> npt.NDArray: ...


@overload
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor: ...


def rotation_6d_to_matrix(d6: Union[np.ndarray, torch.Tensor]):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if isinstance(d6, np.ndarray):
        d6 = torch.from_numpy(d6).float()
        is_numpy = True
    else:
        is_numpy = False

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    matrix = torch.stack((b1, b2, b3), dim=-2)

    if is_numpy:
        matrix = matrix.numpy()

    return matrix


@overload
def matrix_to_rotation_6d(matrix: npt.NDArray) -> npt.NDArray: ...


@overload
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor: ...


def matrix_to_rotation_6d(matrix: Union[np.ndarray, torch.Tensor]):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    if isinstance(matrix, np.ndarray):
        matrix = torch.from_numpy(matrix).float()
        is_numpy = True
    else:
        is_numpy = False

    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
        is_squeeze = True
    else:
        is_squeeze = False

    batch_dim = matrix.size()[:-2]
    rot6d = matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    if is_squeeze:
        rot6d = rot6d.squeeze(0)

    if is_numpy:
        rot6d = rot6d.numpy()

    return rot6d
