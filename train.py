import hydra
import logging
import os
from omegaconf import OmegaConf, DictConfig
from src.algo import *
from src.utils import seed_everything, TensorboardLogger, LoggingSystem, WandbLogger, set_debug_apis
import torch
import torchvision

from src.utils.utils import timer

log = logging.getLogger(__name__)


def create_model(cfg: DictConfig, writer) -> Algo:
    """ Creates the controller for the algorithm of choice and reloads from a checkpoint if requested """
    method: Algo = eval(cfg.algo.classname)(model_info=cfg.model, device=cfg.device, writer=writer,
                                            dataset=cfg.dataset, params=cfg.algo.params,
                                            savedir=cfg.savedir, output_suffix=cfg.output_suffix)
    if cfg.checkpoint_path is not None:
        method.load_from_checkpoint(cfg.checkpoint_path)
    return method

@timer('training_fit', log)
def train_fit(cfg: DictConfig, writer) -> None:
    """ Issues algorithm controller and start the training """
    model: Algo = create_model(cfg, writer)
    if cfg.do_train:
        model.fit(cfg.n_round)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """ Entry-point for any algorithm training

        Sets the root directory, seeds all the number generators, print parameters and
        starts the training.
    """
    OmegaConf.resolve(cfg)
    os.chdir(cfg.root)
    seed_everything(cfg.seed, cfg.debug)
    set_debug_apis(cfg.debug)
    log.info("Parameters:\n" + OmegaConf.to_yaml(cfg))
    writer: LoggingSystem = eval(cfg.logger.classname)(**cfg.logger.params, config=OmegaConf.to_object(cfg))
    train_fit(cfg, writer)
    writer.finish()


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    main()
