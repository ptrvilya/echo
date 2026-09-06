from logging import getLogger
from pathlib import Path
from copy import deepcopy

import torch

from echo.configs import ProjectConfig
from echo.data.batch_data import BatchData
from echo.data.dataset import MotionDataset

logger = getLogger(__name__)


def dataset_name_to_config(dataset_name: str, config: ProjectConfig):
    if dataset_name == "amass":
        dataset_config = config.amass
    elif dataset_name == "behave":
        dataset_config = config.behave
    elif dataset_name == "omomo":
        dataset_config = config.omomo
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name}")

    kwargs = dict()

    if dataset_name in ["behave", "omomo"]:
        # object-specific arguments
        kwargs.update(
            {
                "obj2classid": dataset_config.obj2classid,
                "include_object": dataset_config.include_object,
                "include_fake_object": False,
            }
        )
    else:
        kwargs.update({"include_fake_object": dataset_config.include_fake_object})

    return dataset_config, kwargs


def get_train_dataloaders(config: ProjectConfig):
    # list of all used datasets
    train_datasets, val_datasets = [], []

    # First create datasets with hoi then with motion only
    _hoi_datasets, _motion_datasets = [], []
    for dataset_name in config.run.datasets:
        dataset_config, _ = dataset_name_to_config(dataset_name, config)
        if dataset_config.include_fake_object:
            _motion_datasets.append(dataset_name)
        else:
            _hoi_datasets.append(dataset_name)

    obj2classid, obj_pointnext, obj_keypoints = dict(), dict(), dict()
    for dataset_name in _hoi_datasets + _motion_datasets:
        dataset_config, kwargs = dataset_name_to_config(dataset_name, config)

        train_dataset = MotionDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split="train",
            conditioning_config=config.model_conditioning,
            model_config=config.model,
            assets_folder=Path(config.env.assets_folder),
            split_file=dataset_config.train_split_file,
            slice_strategy="random",
            num_betas=config.model_mesh.num_betas,
            fps=dataset_config.fps_train,
            preload_files=dataset_config.preload_files,
            overlap_size=config.sample.overlap_size,
            **kwargs,
        )
        val_dataset = MotionDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split="test",
            conditioning_config=config.model_conditioning,
            model_config=config.model,
            assets_folder=Path(config.env.assets_folder),
            split_file=dataset_config.test_split_file,
            slice_strategy="deterministic",
            num_betas=config.model_mesh.num_betas,
            fps=dataset_config.fps_eval,
            preload_files=dataset_config.preload_files,
            overlap_size=config.sample.overlap_size,
            **kwargs,
        )

        obj2classid.update(deepcopy(train_dataset.obj2classid))
        obj_pointnext.update(deepcopy(train_dataset.obj_pointnext))
        obj_keypoints.update(deepcopy(train_dataset.obj_keypoints))

        if dataset_name in config.train.datasets:
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        else:
            del train_dataset
            del val_dataset

    for dataset in train_datasets + val_datasets:
        if dataset.name in _motion_datasets:
            dataset.obj2classid = obj2classid
            dataset.obj_pointnext = obj_pointnext

    # concatenate datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    if config.dataloader.sampler == "weighted":
        train_weights = []
        for dataset in train_datasets:
            dataset_config, _ = dataset_name_to_config(dataset.name, config)
            sample_weight = getattr(dataset_config, "sample_weight", 1)

            weights = torch.full(
                (len(dataset),), float(sample_weight), dtype=torch.double
            )
            train_weights.append(weights)
            logger.info(f"Sample weight for {dataset.name}: {sample_weight}")
        train_weights = torch.cat(train_weights, dim=0)
        sampler = torch.utils.data.WeightedRandomSampler(
            train_weights,
            num_samples=config.dataloader.num_samples_per_epoch,
            replacement=True,
        )
    elif config.dataloader.sampler == "random":
        sampler = torch.utils.data.RandomSampler(
            train_dataset, num_samples=config.dataloader.num_samples_per_epoch
        )
    elif config.dataloader.sampler == "default":
        sampler = None
    else:
        raise NotImplementedError(f"Unknown sampler: {config.dataloader.sampler}")

    # create dataloaders
    # keep workers alive across epochs; prefetch deeper to hide loading gaps
    use_workers = config.dataloader.workers > 0
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        num_workers=config.dataloader.workers,
        drop_last=True,
        sampler=sampler,
        pin_memory=True,
        collate_fn=BatchData.collate,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.dataloader.batch_size_val,
        num_workers=config.dataloader.workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=BatchData.collate,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
    )
    logger.info(f"Train data length: {len(train_dataset)}")
    logger.info(f"Val data length: {len(val_dataset)}")

    return train_dataloader, val_dataloader, obj_keypoints


def get_eval_dataloaders(config: ProjectConfig):
    # list of all used datasets
    datasets = []

    # First create datasets with hoi then with motion only
    _hoi_datasets, _motion_datasets = [], []
    run_datasets = set(config.run.datasets)
    run_datasets.update(config.sample.datasets)

    for dataset_name in run_datasets:
        dataset_config, _ = dataset_name_to_config(dataset_name, config)
        if dataset_config.include_fake_object:
            _motion_datasets.append(dataset_name)
        else:
            _hoi_datasets.append(dataset_name)

    obj2classid, obj_pointnext, obj_keypoints = dict(), dict(), dict()
    for dataset_name in _hoi_datasets + _motion_datasets:
        dataset_config, kwargs = dataset_name_to_config(dataset_name, config)

        # Only preload files if dataset is used for sampling
        if (
            len(config.sample.datasets) > 0
            and dataset_config.name in config.sample.datasets
        ):
            preload_files = dataset_config.preload_files
        else:
            preload_files = False

        dataset = MotionDataset(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            split="test",
            assets_folder=Path(config.env.assets_folder),
            conditioning_config=config.model_conditioning,
            model_config=config.model,
            split_file=dataset_config.test_split_file,
            slice_strategy=config.sample.slice_strategy,
            num_betas=config.model_mesh.num_betas,
            fps=dataset_config.fps_eval,
            preload_files=preload_files,
            overlap_size=config.sample.overlap_size,
            noise_simulation_percent=config.sample.noise_simulation_percent,
            **kwargs,
        )

        # accumulate datasets
        datasets.append(dataset)

        obj2classid.update(dataset.obj2classid)
        obj_keypoints.update(dataset.obj_keypoints)
        obj_pointnext.update(dataset.obj_pointnext)

    for dataset in datasets:
        if dataset.name in _motion_datasets:
            dataset.obj2classid = obj2classid
            dataset.obj_pointnext = obj_pointnext

    # create dataloaders
    dataloaders = []
    for dataset in datasets:
        # Only create dataloader if dataset is used for sampling
        if len(config.sample.datasets) > 0 and dataset.name in config.sample.datasets:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=config.dataloader.workers,
                shuffle=False,
                pin_memory=False,
                collate_fn=BatchData.collate,
            )

            dataloaders.append(dataloader)
            logger.info(f"Eval data length for {dataset.name}: {len(dataset)}")

    return dataloaders, obj_keypoints
