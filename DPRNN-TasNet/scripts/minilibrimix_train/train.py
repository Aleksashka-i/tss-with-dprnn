import sys
sys.path.append('../../')

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.datasets.base import BaseDataset
from src.trainers.dprnntasnet_trainer import Trainer

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    logger = logging.getLogger("train")

    OmegaConf.resolve(config)

    logger.info("RUN "+config["name"])
    logger.info("Initializing Dataloader (MiniLibriMix)....")
    trainDataloader, testDataloader = BaseDataset.loaders_from_mini(batch_size=4)
    logger.info("OK")
    logger.info(str(len(trainDataloader))+ " "+ str(len((testDataloader))))
    
    logger.info("Initializing model....")
    model = hydra.utils.instantiate(config["model"])
    logger.info("OK")

    logger.info("Initializing trainer....")
    trainer = Trainer(model, logger, config)
    logger.info("OK")

    logger.info("Initiating trainer run...")
    trainer.run(trainDataloader, testDataloader, config["epochs"])
    logger.info("trainer run COMPLETED")

if __name__ == "__main__": 
    main()