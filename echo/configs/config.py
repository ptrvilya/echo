import os
from dataclasses import dataclass, field

from .datasets import AmassConfig, BehaveConfig, OmomoConfig
from .environment import EnvironmentConfig


@dataclass
class ConditioningModelConfig:
    # Head-mounted device feature
    use_hmd_feature_conditioning: bool = True
    dim_cond_hmd: int = 52  # head: 6 + 3 + 1 + 6, hands: 12 + 6 + 12 + 6
    # Randomly substitute conditioning with zeros
    condition_drop_prob: float = 0.1
    separate_fake_object: bool = True

    use_pointnext_conditioning: bool = True
    dim_cond_pointnext: int = 1024

    use_class_conditioning: bool = True
    dim_cond_class: int = 18
    fourier_encoding_freqs: int = 0

    # Prefix conditioning: clean x0 of the first K frames (+ known-frame mask)
    # appended to the per-frame conditioning so windows can continue the
    # previous window's prediction
    use_prefix_conditioning: bool = False
    prefix_prob: float = 0.5  # share of training samples with a prefix
    prefix_len_max: int = 30  # K ~ U[1, prefix_len_max]
    prefix_noise_std: float = 0.0  # gaussian noise on the training prefix


@dataclass
class DenoisingModelConfig:
    name: str = "hoic_xdit"
    dim_intermediate_embedding: int = 128
    dim_embedding: int = 384
    params: dict = field(default_factory=dict)  # passed as kwargs to DenoisingModel


@dataclass
class EgoHOIModelConfig:
    # Input and output configuration
    dim_motion_feature: int = 6  # input is B x T x J x D
    dim_input_object: int = 9  # input is B x T x D, 6D global rotation + translation
    dim_output_motion: int = 6  # output is B x T x J x D
    dim_output_object: int = 9  # output is B x T x D
    dim_contacts_hoi: int = 64  # 64 - hoi contact, 8 - env contact
    dim_contacts_env: int = 8
    dim_contacts: int = "${eval:'${.dim_contacts_hoi} + ${.dim_contacts_env}'}"
    num_joints: int = 21
    window_size: int = 120  # window_size

    # Contacts
    include_env_contact: bool = False
    include_hoi_contact: bool = False

    # Object
    object_pose: str = "canonical"  # canonical or relative

    # diffusion
    denoise_mode: str = (
        "sample"  # epsilon or sample (as in scheduler - prediction_type)
    )
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = "linear"  # 'custom'

    # classifier guidance
    cg_apply: bool = False
    cg_scale: float = 0.03
    cg_t_stamp: int = 200
    cg_skate_relative_scale: float = 0.2
    cg_contact_threshold: float = 0.05


@dataclass
class MeshModelConfig:
    type: str = "smplx"
    num_betas: int = 10

    include_object: bool = False
    with_skin: bool = False
    inference_chunk_size: int = 32


@dataclass
class RunConfig:
    name: str = "debug"
    job: str = "train"  # train, sample, eval
    cpu: bool = False
    seed: int = 4242
    datasets: list[str] = field(default_factory=list)

    # Overwritten automatically during init
    path: str = os.path.join("${env.experiments_folder}", "${run.name}")


@dataclass
class ResumeConfig:
    checkpoint: str | None = None  # path to the checkpoint
    step: int | None = -1
    use_ema: bool = True  # prefer EMA weights at inference when available
    training: bool = True
    training_optimizer: bool = True
    training_scheduler: bool = True
    training_state: bool = True


@dataclass
class OptimizerConfig:
    type: str = "torch"
    name: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-6
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: float | None = 10.0  # 5.0
    kwargs: dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))


@dataclass
class SchedulerConfig:
    type: str = "transformers"
    name: str = "cosine"
    kwargs: dict = field(
        default_factory=lambda: dict(
            num_warmup_steps=15000,  # 0
            num_training_steps="${train.max_steps}",
        )
    )


@dataclass
class DataloaderConfig:
    batch_size: int = 256  # 2 for debug
    batch_size_val: int = 128  # 2 for debug
    workers: int = 10  # 0 for debug

    sampler: str = "weighted"  # 'default', 'random' or 'weighted'
    num_samples_per_epoch: int = 100000


@dataclass
class TrainConfig:
    mixed_precision: str = "no"  # 'no', 'bf16'
    compile_denoiser: bool = False  # torch.compile the transformer forward
    ema_decay: float | None = None  # e.g. 0.9999; None disables EMA
    limit_train_batches: int | None = None
    limit_val_batches: int | None = 2
    log_step_freq: int = 50
    val_freq: int = 100  # in epochs
    infer_meshes: bool = True  # only needed for contacts and auxillary losses
    checkpoint_freq: int = 10_000
    max_steps: int = 400_000
    loss_t_stamp_threshold: int = 150  # threshold for loss computation

    aux_loss_warmup_steps: int = 50000
    # linear ramp-in of aux losses over this many steps after warmup (0 = off)
    aux_loss_ramp_steps: int = 0

    datasets: list[str] = "${run.datasets}"

    # Losses
    losses: dict[str, float] = field(default_factory=dict)


@dataclass
class SampleConfig:
    scheduler: str = "ddpm_b"
    num_inference_steps: int = 100

    overlap_size: int = 30
    slice_strategy: str = "deterministic"

    alpha: float = 0.0  # 0.4 for smooth inpaint

    # percentage of conditioning to drop
    noise_simulation_percent: int = 0

    # by default will be replaced by run.datasets
    datasets: list[str] = field(default_factory=list)

    artifacts_folder_suffix: str = ""

    mode: str = "sample_naive"
    modalities: str = "11"
    repetition_id: list[int] = field(default_factory=lambda: [1])


@dataclass
class EvalConfig:
    use_rec_metrics: bool = True

    sampling_target: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    wandb: bool = True
    wandb_project: str = "echo"
    wandb_entity: str | None = None  # None: wandb default entity


@dataclass
class ProjectConfig:
    amass: AmassConfig = AmassConfig()
    behave: BehaveConfig = BehaveConfig()
    omomo: OmomoConfig = OmomoConfig()

    model_denoising: DenoisingModelConfig = DenoisingModelConfig()
    model_conditioning: ConditioningModelConfig = ConditioningModelConfig()
    model: EgoHOIModelConfig = EgoHOIModelConfig()
    model_mesh: MeshModelConfig = MeshModelConfig()

    env: EnvironmentConfig = EnvironmentConfig()
    run: RunConfig = RunConfig()
    resume: ResumeConfig = ResumeConfig()

    dataloader: DataloaderConfig = DataloaderConfig()
    train: TrainConfig = TrainConfig()
    logging: LoggingConfig = LoggingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()

    sample: SampleConfig = SampleConfig()
    eval: EvalConfig = EvalConfig()
