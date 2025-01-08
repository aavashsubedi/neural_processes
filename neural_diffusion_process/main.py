import hydra
import omegaconf
import os
from train_mnist import train_mnist
from utils import setup_wandb

@hydra.main(version_base='1.3', config_path="",
             config_name="config.yaml")
def main(cfg):
    setup_wandb(cfg)

    train_mnist(cfg=cfg, dataset_path=cfg.data_dir, run_name=cfg.run_name)
    return 0

if __name__ == "__main__":
    main()