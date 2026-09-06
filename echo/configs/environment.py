import os
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    datasets_folder: str = "./data/preprocessed"
    assets_folder: str = "./assets/"

    experiments_folder: str = "./experiments"
    smpl_all_models_folder: str = "./data/body_models/smplx/models"

    smplx_folder: str = os.path.join("${env.smpl_all_models_folder}", "smplx")
    smplh_amass_folder: str = os.path.join(
        "${env.smpl_all_models_folder}", "smplh_amass"
    )

    # raw dataset roots used by the preprocessing scripts
    amass_folder: str = "./data/raw/amass/"
    behave_folder: str = "./data/raw/behave"
    # output of convert_behave_smplh2x.py (SMPL-H fits converted to SMPL-X)
    behave_converted_folder: str = "./data/raw/behave/smplh2x_smplfitter"
    omomo_folder: str = "./data/raw/omomo/"
    # DATA_ROOT for smplfitter: body models under <folder>/body_models
    smplfitter_data_folder: str = "./data/body_models/smplx"
