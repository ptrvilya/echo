import torch

from echo.configs import ProjectConfig
from echo.core.evaluator import Evaluator
from echo.core.sampler import Sampler
from echo.core.trainer import Trainer
from echo.model import get_model
from echo.utils.exp import init_exp, init_logging, init_wandb, parse_arguments
from echo.utils.training import set_seed


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Parse arguments
    arguments = parse_arguments()

    # Initialise run
    config: ProjectConfig = init_exp(arguments)

    # Set random seed
    set_seed(config.run.seed)

    # Logging
    init_logging(config)
    if config.logging.wandb:
        init_wandb(config)

    if config.run.job in ["train", "sample"]:
        # Create model
        model = get_model(config)

        if config.run.job == "train":
            # Create trainer
            trainer = Trainer(config, model)

            # Run training
            trainer.train()
        elif config.run.job == "sample":
            sampler = Sampler(config, model)
            sampler.sample()
    elif config.run.job == "eval":
        evaluator = Evaluator(config)
        evaluator.evaluate()
    else:
        raise ValueError(f"Invalid job type {config.run.job}")


if __name__ == "__main__":
    main()
