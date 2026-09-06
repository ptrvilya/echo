## SMPL-X model
The project uses the [smplx](https://github.com/vchoutas/smplx) package to work with the SMPL-X body model.
Follow [this](https://github.com/vchoutas/smplx#downloading-the-model) link for the instructions on setting it up.
After installing smplx set `env.smpl_all_models_folder` in `scenarios/env.yaml` to the folder with the
[following](https://github.com/vchoutas/smplx#model-loading) structure (`smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz` are required;
the `smplh/` folder is only needed for the BEHAVE conversion):
```bash
<smpl_all_models_folder>
├── smplx
│   ├── SMPLX_NEUTRAL.npz
│   ├── SMPLX_MALE.npz
│   ├── SMPLX_FEMALE.npz
├── smplh
│   ├── ...
```
All datasets are converted to the neutral SMPL-X body using the shape mappings stored in
`assets/gender_conversion*/` (`common.convert_to_neutral=True`, the default).


## Downloading data
Raw datasets are read from the roots defined in the environment config [scenarios/env.yaml](../scenarios/env.yaml)
(`env.amass_folder`, `env.behave_folder`, `env.omomo_folder`). Either edit the file or override them on the
command line.

### AMASS
1. Download the **SMPL-X N** (neutral) version of the subsets listed below from the official
   [website](https://amass.is.tue.mpg.de/).
   Train: `ACCAD, BMLmovi, BioMotionLab_NTroje, CMU, DFaust_67, DanceDB, EKUT, Eyes_Japan_Dataset, KIT,
   MPI_Limits, TCD_handMocap, TotalCapture`; val: `HumanEva, MPI_HDM05, MPI_mosh, SFU`;
   test: `Transitions_mocap, SSM_synced` (see `amass.subsets_*`).
2. Extract every subset into its own folder under `env.amass_folder`.

#### Folders structure
```bash
<env.amass_folder>
├── ACCAD
│   ├── Female1General_c3d
│   │   ├── A1 - Stand_stageii.npz
│   │   ├── ...
│   ├── ...
├── CMU
├── ...
```

### BEHAVE
1. Download data from the official [website](https://virtualhumans.mpi-inf.mpg.de/behave/license.html).
   Required data: Scanned objects, Annotations at 30fps, and the train/test split (`split.json`).
2. Extract data, following the folder structure below.
3. Convert the SMPL+H fits to SMPL-X. The conversion is done with
   [smplfitter](https://github.com/isarandi/smplfitter), which expects the body models
   under `<env.smplfitter_data_folder>/body_models` (see the smplfitter documentation):
   ```bash
   python -m echo.data_parsers.convert_behave_smplh2x -c scenarios/env.yaml
   ```
   The output is written to `env.behave_converted_folder` and read from there by the preprocessing.

#### Folders structure
```bash
<env.behave_folder>
├── behave-30fps-params-v1
│   ├── Date01_Sub01_backpack_back
│   │   ├── smpl_fit_all.npz
│   │   ├── object_fit_all.npz
│   │   ├── info.json
│   ├── ...
├── objects
│   ├── backpack
│   │   ├── backpack_f1000.ply
│   │   ├── ...
│   ├── ...
├── split.json
<env.behave_converted_folder>            # output of convert_behave_smplh2x.py
├── Date01_Sub01_backpack_back.npz
├── ...
```

### OMOMO
1. Download the dataset from the official [repository](https://github.com/lijiaman/omomo_release).
   Required data: sequence annotations (`train/test_diffusion_manip_seq_joints24.p`) and the captured object
   meshes (`data/captured_objects/*_cleaned_simplified.obj`).
2. The meshes are canonicalized during preprocessing with the original-to-canonical transforms shipped in
   `assets/objects/omomo/canonical_object_transform.pkl` (`(R @ v + t) * scale`).

#### Folders structure
```bash
<env.omomo_folder>
├── data
│   ├── train_diffusion_manip_seq_joints24.p
│   ├── test_diffusion_manip_seq_joints24.p
│   ├── captured_objects
│   │   ├── clothesstand_cleaned_simplified.obj
│   │   ├── ...
```


## Preprocessing data
Preprocessing scripts are located in `echo/data_parsers/`. Before running them update the environment
configuration in `scenarios/env.yaml` (the same file is used for training):
```
env:
  datasets_folder: ./data/preprocessed/   # Folder to store preprocessed datasets
  assets_folder: ./assets/                # Folder with assets (usually ./assets/ within repo)
  experiments_folder: ./experiments       # Folder with experiments (usually ./experiments/ within repo)
  smpl_all_models_folder: ./data/smplx_models/  # Folder with SMPL-X models
  amass_folder: ./data/raw/amass/         # Raw AMASS root
  behave_folder: ./data/raw/behave/       # Raw BEHAVE root
  behave_converted_folder: ./data/raw/behave_smplx/  # Output of convert_behave_smplh2x.py
  omomo_folder: ./data/raw/omomo/         # Raw OMOMO root
  smplfitter_data_folder: ./data/smplfitter/  # smplfitter DATA_ROOT (body_models/ inside)
```

Canonical object transforms, object keypoints and PointNeXt features for BEHAVE and OMOMO are shipped in
`assets/objects/<dataset>/` and are reused by the preprocessing so that the objects match the trained models;
only the canonical meshes are regenerated (`-g` flag) from the raw data.
The simplest way to run everything is the wrapper script
(`DATASETS`, `DATASETS_FOLDER`, `EXTRA_OVERRIDES`, `CLEAN=1`, `BEHAVE_CONVERT=1`, `DRY_RUN=1` env variables):
```bash
DATASETS="amass behave omomo" DATASETS_FOLDER=./data/preprocessed scripts/preprocess.sh
```
Or run the parsers one by one:
```bash
# AMASS
python -m echo.data_parsers.preprocess_amass_smplx -c scenarios/env.yaml -- amass.split=train
python -m echo.data_parsers.preprocess_amass_smplx -c scenarios/env.yaml -- amass.split=val
python -m echo.data_parsers.preprocess_amass_smplx -c scenarios/env.yaml -- amass.split=test

# BEHAVE (-g generates canonical object meshes, needed once)
python -m echo.data_parsers.preprocess_behave -g -c scenarios/env.yaml -- behave.split=train
python -m echo.data_parsers.preprocess_behave -c scenarios/env.yaml -- behave.split=test

# OMOMO
python -m echo.data_parsers.preprocess_omomo -g -c scenarios/env.yaml -- omomo.split=train
python -m echo.data_parsers.preprocess_omomo -c scenarios/env.yaml -- omomo.split=test
```

The resulting structure of `env.datasets_folder` (one `.hdf5` file per sequence with SMPL-X parameters,
motion features, head/wrist tracking features, object poses and contacts):
```bash
<env.datasets_folder>
├── amass
│   ├── dataset_train_30fps
│   │   ├── <sequence>.hdf5
│   │   ├── ...
│   ├── dataset_val_30fps
│   ├── dataset_test_30fps
├── behave
│   ├── dataset_train_30fps
│   ├── dataset_test_30fps
│   ├── object_meshes                    # canonical meshes (.ply)
│   ├── object_keypoints                 # copied from assets/objects/behave
│   ├── canonical_object_transform.pkl   # copied from assets/objects/behave
│   ├── object_pointnext.pkl             # copied from assets/objects/behave
├── omomo
│   ├── ...                              # same layout as behave
```
