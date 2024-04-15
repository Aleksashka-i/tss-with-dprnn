import sys
import logging
import hydra

from omegaconf import DictConfig, OmegaConf

sys.path.append('../..')

from src.datasets.librimix import get_train_dataloader, get_eval_dataloader
from src.trainers.trainer import Trainer
from src.reporters.reporter import Reporter

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    logger = logging.getLogger('train')

    OmegaConf.resolve(config)

    logger.info('RUN %s', config['name'])
    logger.info('Initializing Datasets and Dataloaders (MiniLibriMix)....')
    _, train_dataloader = get_train_dataloader(config)
    val_set, test_dataloader  = get_eval_dataloader(config)
    logger.info('OK')
    logger.info(str(len(train_dataloader) + len(test_dataloader)))
    
    logger.info('Initializing data for reporter....')
    eval_mixtures = {}
    for id_ in config['logs']['metadata']['ids']:
        if id_ >= len(val_set):
            logger.info('Mixture id is out of bound (len of eval_set is {})!'.format(len(val_set)))
            raise ValueError
        mix, sources = val_set[id_]
        eval_mixtures[id_] = {
            'mix': mix,
            's1_target': sources[0],
            's2_target': sources[1],
        }
    if len(val_set) == 0:
        logger.info('No mixtures were added for inference.')
    else:
        logger.info('{} mixtures were added for inference.'.format(config['logs']['metadata']['ids']))
    logger.info('OK')

    logger.info('Initializing reporter....')
    reporter = Reporter(config, logger)
    logger.info('OK')

    logger.info('Initializing model....')
    model = hydra.utils.instantiate(config['model'])
    logger.info('OK')

    logger.info('Initializing trainer....')
    trainer = Trainer(model, logger, eval_mixtures, reporter, config)
    logger.info('OK')

    logger.info('Initiating trainer run...')
    trainer.run(train_dataloader, test_dataloader, config['epochs'], config['early_stop'])
    logger.info('trainer run COMPLETED')

    reporter.wandb_finish()

if __name__ == '__main__':
    main()