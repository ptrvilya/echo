from collections import defaultdict
from logging import getLogger
from pathlib import Path

import numpy as np

from echo.configs import ProjectConfig
from echo.utils.metrics import reconstruction


class Evaluator:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.logger = getLogger(__name__)

    def evaluate(self):
        base_name = "artifacts"
        if len(self.config.sample.artifacts_folder_suffix) > 0:
            base_name = f"{base_name}_{self.config.sample.artifacts_folder_suffix}"
        base_samples_folder = (
            Path(self.config.run.path)
            / base_name
            / f"step_{self.config.resume.step}_samples"
        )
        self.logger.info(
            f"Experiment: {self.config.run.name} step: {self.config.resume.step}"
        )
        self.logger.info(f"Artifacts folder: {base_name}")

        # Reconstruction
        if self.config.eval.use_rec_metrics:
            self.logger.info("Evaluating reconstruction")
            for dataset in self.config.sample.datasets:
                self.logger.info(f"\ton {dataset}")
                dataset_samples_folder = base_samples_folder / f"{dataset}_test"

                # Reconstruction
                metrics = defaultdict(list)
                for sample_target in self.config.eval.sampling_target:
                    self.logger.info(f"\t\tSample target: {sample_target}")
                    repetition_folders = sorted(
                        list(
                            dataset_samples_folder.glob(f"{sample_target}/repetition_*")
                        )
                    )
                    for repetition_folder in repetition_folders:
                        # subject
                        if "sbj" in sample_target:
                            (
                                mpjpe,
                                mpjve,
                                skate,
                                skate_gt,
                                contact,
                                contact_gt,
                                env_contact_metrics,
                            ) = reconstruction.get_sbj_metrics(
                                self.config, repetition_folder, dataset
                            )
                            metrics["MPJPE"].append(mpjpe)
                            metrics["MPJVE"].append(mpjve)
                            metrics["FOOT_SKATE"].append(skate)
                            metrics["FOOT_SKATE_GT"].append(skate_gt)
                            metrics["FOOT_CONTACT"].append(contact)
                            metrics["FOOT_CONTACT_GT"].append(contact_gt)
                            for k, v in env_contact_metrics.items():
                                metrics[f"ENV_CONTACT_{k.upper()}"].append(v)

                        # object
                        if "obj" in sample_target:
                            (
                                mpjpe_obj,
                                obj_c,
                                obj_rot_diff,
                                obj_quat_diff,
                                diff_contact_metrics,
                                mesh_contact_metrics,
                            ) = reconstruction.get_obj_metrics(
                                self.config, repetition_folder, dataset
                            )
                            metrics["MPJPE_OBJ"].append(mpjpe_obj)
                            metrics["OBJ_CENTER"].append(obj_c)
                            metrics["OBJ_ROT_DIFF"].append(obj_rot_diff)
                            metrics["OBJ_QUAT_DIFF"].append(obj_quat_diff)
                            for k, v in diff_contact_metrics.items():
                                metrics[f"DIFF_CONTACT_{k.upper()}"].append(v)
                            for k, v in mesh_contact_metrics.items():
                                metrics[f"MESH_CONTACT_{k.upper()}"].append(v)

                for k, v in metrics.items():
                    if np.array(v).size > 0:
                        v = np.stack(v, 1)
                        self.logger.info(
                            f"\t\t{k:<12s}: {np.nanmean(v):.4f} ± {np.nanmean(np.std(v, 1)):.4f}"
                        )

                s = ""
                for k in [
                    "MPJPE",
                    "MPJVE",
                    "FOOT_CONTACT",
                    "FOOT_SKATE",
                    "MPJPE_OBJ",
                    "OBJ_CENTER",
                    "OBJ_ROT_DIFF",
                    "OBJ_QUAT_DIFF",
                ]:
                    v = metrics[k]
                    if len(v) > 0:
                        v = np.stack(v, 1)
                        mn = np.nanmean(v)
                        std = np.nanmean(np.nanstd(v, 1))
                        s += f"{mn:.02f} ± {std:.02f};"
                self.logger.info(s)

                # Contacts separately
                s = ""
                for k in metrics.keys():
                    if "ENV_CONTACT" in k:
                        v = metrics[k]
                        if len(v) > 0:
                            v = np.stack(v, 1)
                            mn = np.nanmean(v)
                            std = np.nanmean(np.nanstd(v, 1))
                            s += f"{mn:.02f} ± {std:.02f};"
                self.logger.info("ENV contact: " + s)
                s = ""
                for k in metrics.keys():
                    if "DIFF_CONTACT" in k or "MESH_CONTACT" in k:
                        v = metrics[k]
                        if len(v) > 0:
                            v = np.stack(v, 1)
                            mn = np.nanmean(v)
                            std = np.nanmean(np.nanstd(v, 1))
                            s += f"{mn:.02f} ± {std:.02f};"
                self.logger.info("HOI contact: " + s)
