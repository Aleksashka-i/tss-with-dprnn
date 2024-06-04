import sys
import logging
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('../../')

# Define a function to handle imports based on the mode
def import_classes(mode):
    if mode == 'bss':
        from src.datasets.librimix import get_train_dataloader, get_eval_dataloader
        from src.trainers.trainer import Trainer
        return get_train_dataloader, get_eval_dataloader, Trainer, None
    
    from src.datasets.librimix_spe import get_train_spe_dataloader, get_eval_spe_dataloader
    from src.reporters.reporter import Reporter
    
    if mode == 'tss_spe':
        from src.trainers.trainer_spe import TrainerSpe
        return get_train_spe_dataloader, get_eval_spe_dataloader, TrainerSpe, Reporter
    elif mode == 'tss_rawnet':
        from src.trainers.trainer_rawnet import TrainerRawNet
        return get_train_spe_dataloader, get_eval_spe_dataloader, TrainerRawNet, Reporter
    else:
        raise ValueError(f"Invalid mode: {mode}")

@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='bss', choices=['bss', 'tss_spe', 'tss_rawnet'], help='Specify the mode to run: bss, tss_spe, or tss_rawnet')
    args = parser.parse_args()
    
    mode = args.mode
    get_train_dataloader, get_eval_dataloader, TrainerClass, ReporterClass = import_classes(mode)

    logger = logging.getLogger('train')

    OmegaConf.resolve(config)

    logger.info('RUN %s', config['name'])
    logger.info('Initializing Datasets and Dataloaders....')
    _, train_dataloader = get_train_dataloader(config)
    eval_set, eval_dataloader = get_eval_dataloader(config)
    logger.info('OK')
    logger.info('train dataloader len: {}'.format(str(len(train_dataloader))))
    logger.info('test dataloader len: {}'.format(str(len(eval_dataloader))))

    logger.info('Initializing data for reporter....')
    eval_mixtures = {}
    for id_ in config['logs']['metadata']['ids']:
        if id_ >= len(eval_set):
            logger.info('Mixture id is out of bound (len of eval_set is {})!'.format(len(eval_set)))
            raise ValueError
        if mode == 'bss':
            mix, sources = eval_set[id_]
            eval_mixtures[id_] = {
                'mix': mix,
                's1_target': sources[0],
                's2_target': sources[1],
            }
        else:
            mix, target, reference, _ = eval_set[id_]
            eval_mixtures[id_] = {
                'mix': mix,
                'target': target,
                'reference': reference,
            }
    if len(eval_set) == 0:
        logger.info('No mixtures were added for inference.')
    else:
        logger.info('{} mixtures were added for inference.'.format(
            len(config['logs']['metadata']['ids']))
        )
    logger.info('OK')

    logger.info('Initializing reporter....')
    reporter = None
    if ReporterClass:
        reporter = ReporterClass(config, logger)
        logger.info('OK')

    logger.info('Initializing model....')
    model = hydra.utils.instantiate(config['model'])
    logger.info('OK')

    logger.info('Initializing trainer....')
    trainer = TrainerClass(model, logger, eval_mixtures, reporter, config)
    logger.info('OK')

    logger.info('Initiating trainer run...')
    trainer.run(train_dataloader, eval_dataloader, config['epochs'], config['early_stop'])
    logger.info('trainer run COMPLETED')

    if reporter:
        reporter.wandb_finish()

if __name__ == '__main__':
    main()