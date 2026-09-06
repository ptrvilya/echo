# ECHO: Project Structure and Main Concepts

Structural overview of the **ECHO** repository for language models and new developers.

## 1. High-Level Overview
ECHO models 3D Human-Object Interaction (HOI) from egocentric head and hand tracking. Motion capture and interaction datasets (AMASS, BEHAVE, OMOMO) are standardized into a unified HDF5 format with the neutral SMPL-X body model, and a conditional diffusion model synthesizes body motion, object pose, and contacts.

Entry point: `main.py` dispatches `run.job` to `train`, `sample`, or `eval`.
Launch: `python main.py -c scenarios/env.yaml scenarios/echo.yaml -- key=value ...`

## 2. Directory Structure

### `echo/` - Core Python Package
- **`echo/configs/`**: Dataclass configs composed with `OmegaConf`. `config.py` holds `ProjectConfig` (model, conditioning, denoising, train, sample, resume, dataloader). `datasets.py` per-dataset configs, `environment.py` local paths, `preprocessing.py` preprocessing options.
- **`echo/core/`**: Job drivers. `trainer.py` (training loop, losses, validation, checkpoints), `sampler.py` (windowed inference and HDF5 artifact export), `evaluator.py` (metrics over exported samples).
- **`echo/data/`**: `dataset.py` builds windows from the HDF5 files (per-frame Z-reset canonicalization, HMD/hand conditioning features, object pose, contacts). `batch_data.py` is the `BatchData` container. `normalization.py` normalizes object 6D poses.
- **`echo/data_parsers/`**: Standalone preprocessing into the unified HDF5 format: `preprocess_amass_smplx.py`, `preprocess_behave.py`, `preprocess_omomo.py`, `convert_behave_smplh2x.py` (SMPL-H to SMPL-X), and `common.py` (canonical object transforms, keypoints, contacts, motion features). Driven by `scripts/preprocess.sh`.
- **`echo/model/`**:
  - `model_hoic.py`: `EgoHOICModel`, the diffusion model with separate timesteps for motion, object, and contacts, plus classifier guidance (`cg_*` config) on contacts and foot skate.
  - `base.py`: scheduler setup and shared model plumbing.
  - `conditioning/model.py`: HMD and hand features, PointNeXt object features, class conditioning, optional prefix conditioning; conditioning dropout for robustness.
  - `denoising/transformer/xdit_hoic.py`: DiT-style transformer denoiser.
  - `schedulers/ddpm_batched.py`: DDPM scheduler with per-sample timesteps (`sample.scheduler=ddpm_b`).
  - `inference/`: windowed inference strategies selected by `sample.mode`: `naive.py`, `stitching.py`, `inpainting.py`.
  - `mesh_model.py`: SMPL-X forward pass (joints only or with skin) and object keypoint/vertex posing.
  - `output.py`: `EgoModelOutput`, parsing the network output vector.
- **`echo/utils/`**: `exp.py` (config merging, experiment folders, checkpoint resume, wandb), `training.py` (optimizer, scheduler, EMA, checkpoint loading), `rotation.py`, `smpl.py`, `contacts.py`, `metrics/reconstruction.py`, `transforms/` (Lie groups `SO3`, `SE3`).

### `scenarios/` - Configs
`env.yaml` local paths, `echo.yaml` the default model and training scenario. Experiments are created under `env.experiments_folder` with an auto-numbered prefix and store their resolved `config.yaml`.

### `scripts/`
`preprocess.sh` runs the data parsers.

### `assets/`, `docs/`
`assets/objects` holds canonical object transforms, keypoints, and PointNeXt features; `assets/gender_conversion*` the neutral-shape mappings. `docs/data.md` documents data setup.

## 3. Model Overview
- **Diffusion:** DDPM over windows of motion. Motion, object, and contact channels get independent timesteps so each modality can be denoised or given as clean conditioning (`sample.modalities`).
- **State vector:** SMPL-X body pose in 6D rotation, object pose (6D rotation + translation, canonical or head-relative), per-vertex HOI contactness scores and environment (floor) contacts.
- **Conditioning:** per-frame head and hand features (relative transforms, canonicalized rotations, head height), object PointNeXt features and class, optional clean prefix from the previous window. Conditioning is randomly dropped during training.
- **Guidance:** classifier guidance with gradients of contact and foot-skate losses taken w.r.t. `x_t`, enabled by `model.cg_apply`.
- **Losses:** denoising losses per modality plus auxiliary joint, skate, object smoothness, and velocity terms, ramped in after `train.aux_loss_warmup_steps`.

## 4. Key Concepts & Technologies
- **Body model:** neutral SMPL-X everywhere at runtime; SMPL-H only appears in the BEHAVE conversion.
- **Data format:** HDF5 sequences with aligned SMPL-X parameters, head and hand transforms, object poses, and contacts.
- **3D math:** use the `SO3`/`SE3` classes in `echo/utils/transforms` for rigid transforms; `echo/utils/rotation.py` for representation conversions.
- **Experiment management:** `OmegaConf` structured configs with YAML scenarios and CLI overrides; `wandb` logging.

## 5. LLM Assistant Guidelines
1. **Config:** access hyperparameters through the `OmegaConf` config objects, not `argparse`.
2. **Transforms:** use `SO3`/`SE3` for relative poses instead of raw matrix products.
3. **Data:** assume the HDF5 format above; regenerate assets via the data parsers, not by hand.
4. **Resume:** resuming by `run.name` restores the run's saved `config.yaml` over the scenario; only CLI overrides win.
