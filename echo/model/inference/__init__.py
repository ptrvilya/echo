from echo.configs import ProjectConfig

from .inpainting import InferenceInpainter
from .naive import InferenceModelNaive
from .stitching import InferenceStitcher


def get_inference_model(config: ProjectConfig) -> InferenceModelNaive:
    if config.sample.mode == "sample_naive":
        model = InferenceModelNaive(
            window_size=config.model.window_size,
            num_inference_steps=config.sample.num_inference_steps,
            object_pose=config.model.object_pose,
        )
    elif config.sample.mode == "sample_stitch":
        model = InferenceStitcher(
            window_size=config.model.window_size,
            overlap_size=config.sample.overlap_size,
            num_joints=config.model.num_joints,
            num_inference_steps=config.sample.num_inference_steps,
            dim_contacts=config.model.dim_contacts,
            object_pose=config.model.object_pose,
        )
    elif config.sample.mode == "sample_inpaint":
        model = InferenceInpainter(
            window_size=config.model.window_size,
            overlap_size=config.sample.overlap_size,
            num_joints=config.model.num_joints,
            num_inference_steps=config.sample.num_inference_steps,
            dim_contacts=config.model.dim_contacts,
            object_pose=config.model.object_pose,
            alpha=config.sample.alpha,
            prefix_conditioning=config.model_conditioning.use_prefix_conditioning,
        )
    else:
        raise ValueError(f"Unknown inference mode: {config.sample.mode}")

    return model
