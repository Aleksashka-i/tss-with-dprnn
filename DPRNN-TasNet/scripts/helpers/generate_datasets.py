import sys
import logging
import hydra
import pickle as pkl

from omegaconf import DictConfig, OmegaConf

sys.path.append('../../')
from src.datasets.librimix import Librimix

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    logger = logging.getLogger('test')

    OmegaConf.resolve(config)

    logger.info('RUN %s', config['name'])
    logger.info('Initializing Datasets....')
    train_set = Librimix(
        csv_path=config['data']['train_path'],
        sample_rate=config['sample_rate'],
        nrows=config['data']['nrows_train'],
        segment=config['data']['segment'],
    )
    logger.info('train len {}'.format(len(train_set)))
    eval_set = Librimix(
        csv_path=config['data']['eval_path'],
        sample_rate=config['sample_rate'],
        nrows=config['data']['nrows_eval'],
        segment=config['data']['segment'],
    )
    logger.info('eval len {}'.format(len(eval_set)))
    test_set = Librimix(
        csv_path=config['data']['test_path'],
        sample_rate=config['sample_rate'],
        nrows=config['data']['nrows_test'],
        segment=None,
    )
    logger.info('test len {}'.format(len(test_set)))
    logger.info('OK')

    logger.info('Saving datasets....')
    with open(config['save_path']['train'], 'wb') as f:
        pkl.dump(train_set, f)
    with open(config['save_path']['eval'], 'wb') as f:
        pkl.dump(eval_set, f)
    with open(config['save_path']['test'], 'wb') as f:
        pkl.dump(test_set, f)
    logger.info('OK')

if __name__ == '__main__':
    main()