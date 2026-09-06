import argparse
import dataclasses
import logging
import os
import re
import sys
import time
from logging import getLogger
from pathlib import Path
from typing import Union

import wandb
from omegaconf import OmegaConf

from echo.configs import ProjectConfig

OmegaConf.register_new_resolver("eval", eval)


def _flatten_config(cfg) -> dict:
    flat = {}

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        else:
            flat[prefix[:-1]] = node

    walk(OmegaConf.to_container(cfg, resolve=True))
    return flat


def _drop_unknown_keys(previous, schema, prefix=""):
    """Remove keys from a saved experiment config that no longer exist in the
    structured config (fields removed since the run was started). Plain dict
    fields (e.g. train.losses) have no fixed schema and are kept as is."""
    dropped = []
    if not dataclasses.is_dataclass(OmegaConf.get_type(schema)):
        return dropped
    for key in list(previous.keys()):
        if key not in schema:
            dropped.append(f"{prefix}{key}")
            del previous[key]
        elif OmegaConf.is_dict(previous[key]) and OmegaConf.is_dict(schema[key]):
            dropped += _drop_unknown_keys(
                previous[key], schema[key], prefix=f"{prefix}{key}."
            )
    return dropped


def parse_arguments():
    # Setup parser
    parser = argparse.ArgumentParser(
        "ECHO arguments."
        "To specify overrides for default config and config file type -- before them, e.g.:"
        "python main.py --config config.yaml -- run.name=test_run run.job=train"
        "[Override scheme: default -> config file -> cli]"
    )
    parser.add_argument(
        "--config", "-c", type=str, nargs="+", help="Path to YAML config(-s) file."
    )
    parser.add_argument(
        "overrides", type=str, nargs="*", help="Overrides for the config."
    )

    # Parse arguments
    arguments = parser.parse_args()
    return arguments


def init_exp(arguments):
    # create default config
    structured_config = ProjectConfig()

    # convert to omegaconf
    config = OmegaConf.structured(structured_config)

    # merge with config(-s) from file
    if arguments.config is not None:
        for config_file in arguments.config:
            config = OmegaConf.merge(config, OmegaConf.load(config_file))

    # merge with cli
    if len(arguments.overrides) > 0:
        cli_config = OmegaConf.from_dotlist(arguments.overrides)
        # merge only checkpoint resume details, merge everything else later
        if "resume" in cli_config:
            config.resume = OmegaConf.merge(config.resume, cli_config.resume)
            del cli_config.resume
        if "run" in cli_config and "name" in cli_config.run:
            config.run.name = cli_config.run.name
            del cli_config.run.name
    else:
        cli_config = None

    # Fix absolute path
    config.env.experiments_folder = str(Path(config.env.experiments_folder).resolve())

    # Find experiment folder by prefix (or create a new one)
    experiment_path = get_experiment_by_prefix(
        config.env.experiments_folder, config.run.name
    )

    # determine if and how to continue previous experiment
    if config.resume.checkpoint is not None:  # load from file and start new experiment
        checkpoint_path = Path(config.resume.checkpoint)
        if checkpoint_path.is_file():
            # Start new experiment from existing checkpoint
            # Note: config.resume.step is ignored

            # merge the rest of the cli config
            if cli_config is not None:
                config = OmegaConf.merge(config, cli_config)
        else:
            raise RuntimeError(f"Checkpoint {checkpoint_path} does not exist.")
    # treat config.run.name as prefix
    elif experiment_path.exists():  # Continue previous experiment
        if config.resume.step == -1:
            checkpoint_path = sorted(
                experiment_path.glob("checkpoints/checkpoint-step-*.pth")
            )
            if len(checkpoint_path) > 0:
                checkpoint_path = checkpoint_path[-1]
                config.resume.step = checkpoint_path.stem.split("-")[-1]
            else:
                checkpoint_path = (
                    experiment_path / "checkpoints/checkpoint-step-0000000.pth"
                )
        else:
            checkpoint_path = (
                experiment_path
                / f"checkpoints/checkpoint-step-{config.resume.step:07d}.pth"
            )

        if not checkpoint_path.exists():
            # no checkpoint found, start from scratch
            if cli_config is not None:
                config = OmegaConf.merge(config, cli_config)
            config.resume.checkpoint = None
        else:
            # restore config from the experiment
            previous_config = OmegaConf.load(experiment_path / "config.yaml")
            del previous_config.resume
            for k in _drop_unknown_keys(previous_config, config):
                print(f"WARNING: dropped {k} from the previous experiment config")

            # scenario/CLI values that the restored experiment config will
            # override; surfaced below because this is easy to miss on resume
            intended = (
                OmegaConf.merge(config, cli_config)
                if cli_config is not None
                else config
            )
            intended_flat = _flatten_config(intended)

            if cli_config is not None:
                previous_config = OmegaConf.merge(
                    previous_config, cli_config
                )  # prioritize cli_config
            config = OmegaConf.merge(config, previous_config)

            final_flat = _flatten_config(config)
            overridden = {
                k: (v, final_flat.get(k))
                for k, v in intended_flat.items()
                if final_flat.get(k) != v
            }
            for k, (v, final_v) in sorted(overridden.items()):
                print(
                    f"WARNING: resume kept {k}={final_v} from the previous "
                    f"experiment config (scenario/CLI has {v}); pass it as a "
                    "CLI override to change it"
                )

            # Overwrite checkpoint path with an actual path to the file
            config.resume.checkpoint = str(checkpoint_path.resolve())
    else:  # Start new experiment
        # merge the rest of the cli config
        if cli_config is not None:
            config = OmegaConf.merge(config, cli_config)

        config.resume.checkpoint = None

    # Create nescessary subdirectories
    experiment_path.mkdir(parents=True, exist_ok=True)
    (experiment_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_path / "artifacts").mkdir(parents=True, exist_ok=True)

    # Save experiment path and full name
    config.run.name = experiment_path.name

    # Turn off wandb for evaluation
    if config.run.job != "train":
        config.logging.wandb = False

    OmegaConf.resolve(config)

    # export config to file
    if config.run.job == "train":
        target_path = experiment_path / "config.yaml"
    else:
        target_path = experiment_path / "sample_config.yaml"

    if target_path.is_file():
        os.remove(target_path)

    OmegaConf.save(config=config, f=str(target_path))
    return config


def init_logging(cfg: ProjectConfig):
    # should only be called if accelerator.is_main_process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(Path(cfg.run.path) / f"log_{cfg.run.job}", mode="a"),
            logging.StreamHandler(),
        ],
    )


def get_script_path() -> str:
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_next_experiment_number(exp_root: Union[Path, str]) -> int:
    # naming pattern: {number:04d}_{name:s}
    exp_root = Path(exp_root)
    exps = sorted(exp_root.glob("???_*"))
    if len(exps) == 0:
        exp_number = 0
    else:
        exp_number = int(str(exps[-1].name).split("_")[0]) + 1

    return exp_number


def get_experiment_by_prefix(exp_root: Union[Path, str], prefix: str) -> Path:
    exp_root = Path(exp_root)
    exps = list(exp_root.glob(f"{prefix}*"))
    print(exp_root, prefix, exps)
    if len(exps) == 1:
        return exps[0]
    else:
        if len(exps) == 0:
            if re.match(r"^[0-9]{3}_", prefix):  # already has exp number
                return exp_root / prefix
            else:
                experiment_number = get_next_experiment_number(exp_root)
                return exp_root / f"{experiment_number:03d}_{prefix}"
        else:
            raise RuntimeError(f"Found {len(exps)} experiments for prefix {prefix}")


def init_wandb(cfg: ProjectConfig):
    logger = getLogger(__name__)
    # init wandb, give multiple tries
    initialized = False

    logger.info("Initializing wandb...")
    for _try, init_method in enumerate(["fork", "thread"]):
        try:
            logger.info(f"Trying to initialize WANDB try {_try}")
            wandb.init(
                project=cfg.logging.wandb_project,
                name=cfg.run.name,
                dir=str(cfg.run.path),
                resume=None,
                entity=cfg.logging.wandb_entity,
                job_type=cfg.run.job,
                config=OmegaConf.to_container(cfg),
                settings=wandb.Settings(start_method=init_method),
            )
            wandb.run.log_code(
                root=get_script_path(),
                include_fn=lambda p: any(
                    p.endswith(ext)
                    for ext in (".py", ".json", ".yaml", ".md", ".txt.", ".gin")
                ),
                exclude_fn=lambda p: any(
                    s in p
                    for s in (
                        "output",
                        "experiments",
                        "tmp",
                        "wandb",
                        ".git",
                        ".vscode",
                    )
                ),
            )

            initialized = True
            break
        except Exception as exc:
            logger.error(f"Exception {exc}")
            time.sleep(10)

    if initialized:
        logger.info("Initialized WANDB")
    else:
        raise RuntimeError("Unable to initialize WANDB")
