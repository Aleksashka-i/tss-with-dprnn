import logging
import hydra

from omegaconf import DictConfig, OmegaConf

from src.datasets.librimix import Librimix
from src.inferencers.inferencer import Inferencer

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    logger = logging.getLogger('test')

    OmegaConf.resolve(config)

    logger.info('RUN %s', config['name'])
    logger.info('Initializing Dataset....')
    test_set = Librimix(
        csv_path=config['data']['test_path'],
        sample_rate=config['sample_rate'],
        nrows=config['data']['nrows_test'],
        segment=config['data']['segment'],
    )
    logger.info('OK')
    logger.info(str(len(test_set)))

    logger.info('Initializing model....')
    model = hydra.utils.instantiate(config['model'])
    logger.info('OK')

    logger.info('Initializing inferencer....')
    inferencer = Inferencer(model, logger, config)
    logger.info('OK')

    logger.info('Initiating inferencer run...')
    inferencer.run(test_set)
    logger.info('inferencer run COMPLETED')

if __name__ == '__main__':
    main()
