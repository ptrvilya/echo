import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DatasetConfig:
    name: str
    root: str
    train_split_file: Optional[str]
    test_split_file: Optional[str]

    include_object: bool = False
    fps_train: int = 30
    fps_eval: int = 30
    preload_files: bool = False
    include_fake_object: bool = False


@dataclass
class AmassConfig(DatasetConfig):
    # fields should match the fields in HOIDataset class
    name: str = "amass"
    root: str = os.path.join("${env.datasets_folder}", "amass")
    train_split_file: Optional[str] = os.path.join(
        "${env.assets_folder}", "amass_train.json"
    )
    test_split_file: Optional[str] = os.path.join(
        "${env.assets_folder}", "amass_test.json"
    )


@dataclass
class BehaveConfig(DatasetConfig):
    name: str = "behave"
    root: str = os.path.join("${env.datasets_folder}", "behave")

    sample_weight: int = 50
    fps_train: int = 30
    fps_eval: int = 30
    include_object: bool = True
    train_split_file: Optional[str] = None
    test_split_file: Optional[str] = None

    obj2classid: Dict[str, int] = field(default_factory=lambda: {})


@dataclass
class OmomoConfig(DatasetConfig):
    name: str = "omomo"

    root: str = os.path.join("${env.datasets_folder}", "omomo")

    sample_weight: int = 1
    fps_train: int = 30
    fps_eval: int = 30

    train_split_file: Optional[str] = None
    test_split_file: Optional[str] = None

    include_object: bool = True
    obj2classid: Dict[str, int] = field(default_factory=lambda: {})
