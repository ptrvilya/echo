import os
from dataclasses import dataclass, field

from echo.configs.environment import EnvironmentConfig


@dataclass
class PreprocessAmassConfig:
    root: str = "${env.amass_folder}"

    target_fps: int = 30

    # Split
    split: str = "train"
    subsets_train: list = field(
        default_factory=lambda: [
            "ACCAD",
            "BMLmovi",
            "BioMotionLab_NTroje",
            "CMU",
            "DFaust_67",
            "DanceDB",
            "EKUT",
            "Eyes_Japan_Dataset",
            "KIT",
            "MPI_Limits",
            "TCD_handMocap",
            "TotalCapture",
        ]
    )

    subsets_val: list = field(
        default_factory=lambda: [
            "HumanEva",
            "MPI_HDM05",
            "MPI_mosh",
            "SFU",
        ]
    )

    subsets_test: list = field(
        default_factory=lambda: [
            "Transitions_mocap",
            "SSM_synced",
        ]
    )


@dataclass
class PreprocessBEHAVEConfig:
    root: str = "${env.behave_folder}"
    root_converted: str = "${env.behave_converted_folder}"
    original_objects_path: str = os.path.join("${behave.root}", "objects")
    full_anno_path: str = os.path.join("${behave.root}", "behave-30fps-params-v1")

    target_fps: int = 30

    split: str = "train"
    contact_threshold: float = 0.05
    split_file: str = os.path.join("${behave.root}", "split.json")
    # odd Savitzky-Golay window (frames) smoothing object pose, 0 disables;
    # matches the body smoothing applied by convert_behave_smplh2x.py
    object_smoothing_window: int = 9

    # SMPL-H -> SMPL-X conversion via smplfitter (convert_behave_smplh2x.py)
    smplfitter_output: str = "${env.behave_converted_folder}"
    # DATA_ROOT for smplfitter: body models and correspondence pkl under <root>/body_models
    smplfitter_data_root: str = "${env.smplfitter_data_folder}"
    smplfitter_num_iter: int = 3
    smplfitter_batch_size: int = 500
    # frame subsample cap for the shared-betas shape fit
    smplfitter_max_shape_frames: int = 500
    smplfitter_skip_existing: bool = True
    # odd Savitzky-Golay window (frames) smoothing body pose + transl, 0 disables
    smplfitter_smoothing_window: int = 0
    # optional sequence name filter, empty = all
    smplfitter_sequences: list[str] = field(default_factory=list)


@dataclass
class PreprocessOMOMOConfig:
    root: str = "${env.omomo_folder}"
    target_fps: int = 30
    split: str = "train"
    anno_path: str = os.path.join(
        "${omomo.root}", "data", "${omomo.split}_diffusion_manip_seq_joints24.p"
    )
    raw_objects_path: str = os.path.join(
        "${omomo.root}", "data", "captured_objects"
    )
    canonical_transform_path: str = os.path.join(
        "${omomo.root}", "tools", "orig2can.pkl"
    )

    object_name_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "monitor": "screen",
            "suitcase": "luggage",
        }
    )

    contact_threshold: float = 0.1
    subjects: list[str] = field(default_factory=list)


@dataclass
class PreprocessCommonParamsConfig:
    n_keypoints: int = 1500  # number of points to sample on the object's surfcace
    # per-dataset canonical transforms, object keypoints and pointnext
    # features that must stay fixed across preprocessing runs
    objects_assets_folder: str = os.path.join("${env.assets_folder}", "objects")
    # max distance (m) of stored keypoints to the regenerated mesh surface when
    # exact barycentric reconstruction is not applicable
    keypoints_surface_tolerance: float = 0.012

    convert_to_neutral: bool = True


@dataclass
class PreprocessConfig:
    # environment
    env: EnvironmentConfig = EnvironmentConfig()

    # Common parameters
    common: PreprocessCommonParamsConfig = PreprocessCommonParamsConfig()

    # datasets
    amass: PreprocessAmassConfig = PreprocessAmassConfig()
    behave: PreprocessBEHAVEConfig = PreprocessBEHAVEConfig()
    omomo: PreprocessOMOMOConfig = PreprocessOMOMOConfig()
