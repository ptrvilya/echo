import torch
from torch import Tensor


def sigmoid_decay(x, threshold, sharpness=100):
    """
    Computes a 'soft' inverted sigmoid.
    Output is ~1.0 when x << threshold, 0.5 when x == threshold, and ~0.0 when x >> threshold.

    Args:
        x: Input tensor (distances in m or velocities in m/frame).
        threshold (tau): The value where prob is 0.5.
        sharpness (alpha): Controls steepness.
    """
    # The formula: 1 / (1 + exp(alpha * (x - tau)))
    # We use torch.sigmoid logic for stability: sigmoid(-alpha * (x - tau))
    return torch.sigmoid(-sharpness * (x - threshold))


def compute_env_contactness(
    contact_env: Tensor,
    contact_vel_thresh: float = 0.01,  # 0.015
    contact_toe_height_thresh: float = 0.04,  # if static toe above this height
    contact_ankle_height_thresh: float = 0.08,
    sharpness: float = 100,
):
    """
    Computes continuous contactness score (0.0 to 1.0) instead of binary flags.
    """
    # contact_env is T x 8 x 2: time x (body parts) x (vel, height)
    velocities_env = contact_env[:, :, 0]
    heights_env = contact_env[:, :, 1]

    # 1. Compute 'Stillness' Score (is the foot stoped?)
    # If vel == vel_thresh, score is 0.5
    c_vel = sigmoid_decay(velocities_env, contact_vel_thresh, sharpness)

    # 2. Compute 'Proximity' Score (is the foot close?)
    # If height == height_thresh, score is 0.5
    # per-joint thresholds in ENV_CONTACT_JOINTS order:
    # knees and ankles use the ankle threshold, feet and wrists the toe one
    contact_height_thresh = torch.tensor(
        2 * [contact_ankle_height_thresh]  # knees
        + 2 * [contact_ankle_height_thresh]  # ankles
        + 2 * [contact_toe_height_thresh]  # feet
        + 2 * [contact_toe_height_thresh],  # wrists
        device=heights_env.device,
    ).unsqueeze(0)  # 1 x 8
    c_height = sigmoid_decay(heights_env, contact_height_thresh, sharpness)

    # 3. Combine (Soft AND)
    # Both must be true for valid floor contact.
    contactness = c_vel * c_height

    return contactness


def compute_hoi_contactness(
    contact_hoi: Tensor,
    contact_dist_thresh: float = 0.05,
    sharpness: float = 100,
):
    """
    Computes continuous contactness score (0.0 to 1.0) for human-object interaction.
    """

    contactness = sigmoid_decay(contact_hoi, contact_dist_thresh, sharpness)

    return contactness
