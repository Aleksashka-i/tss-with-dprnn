import sys
import logging
import pickle as pkl

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('../../')
from src.datasets.librimix_spe import LibrimixSpe
from src.inferencers.inferencer_spe import InferencerSpe
from src.reporters.reporter import Reporter

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    logger = logging.getLogger('test')

    OmegaConf.resolve(config)

    logger.info('RUN %s', config['name'])
    logger.info('Initializing Dataset....')
    if config['data']['use_generated_test'] is not None:
        with open(config['data']['use_generated_test'], 'rb') as file:
            test_set = pkl.load(file)
    else:
        test_set = LibrimixSpe(
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

    logger.info('Initializing reporter....')
    reporter = Reporter(config, logger)
    logger.info('OK')

    logger.info('Initializing inferencer....')
    inferencer = InferencerSpe(model, logger, config, reporter)
    logger.info('OK')

    logger.info('Initiating inferencer run...')
    inferencer.run(test_set)
    logger.info('inferencer run COMPLETED')

    reporter.wandb_finish()

if __name__ == '__main__':
    main()
