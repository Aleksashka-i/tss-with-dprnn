import sys
import logging
import pickle as pkl
import argparse

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append('../../')

def import_classes(mode):
    if mode == 'bss':
        from src.datasets.librimix import Librimix
        from src.inferencers.inferencer import Inferencer
        return Librimix, Inferencer, None
    
    from src.datasets.librimix_spe import LibrimixSpe
    from src.reporters.reporter import Reporter
    
    if mode == 'tss_spe':
        from src.inferencers.inferencer_spe import InferencerSpe
        return LibrimixSpe, InferencerSpe, Reporter
    elif mode == 'tss_rawnet':
        from src.inferencers.inferencer_rawnet import InferencerRawNet
        return LibrimixSpe, InferencerRawNet, Reporter
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
@hydra.main(version_base=None, config_path='./', config_name='config')
def main(config: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='tss', choices=['bss', 'tss'], help='Specify the mode to run: default or spe')
    args = parser.parse_args()
    
    mode = args.mode
    DatasetClass, InferencerClass, ReporterClass = import_classes(mode)

    logger = logging.getLogger('test')

    OmegaConf.resolve(config)

    logger.info('RUN %s', config['name'])
    logger.info('Initializing Dataset....')
    if config['data']['use_generated_test'] is not None:
        with open(config['data']['use_generated_test'], 'rb') as file:
            test_set = pkl.load(file)
    else:
        test_set = DatasetClass(
            csv_path=config['data']['test_path'],
            sample_rate=config['data']['sample_rate'],
            nrows=config['data']['nrows_test'],
            segment=config['data']['segment'],
        )
    logger.info('OK')
    logger.info(str(len(test_set)))

    logger.info('Initializing model....')
    model = hydra.utils.instantiate(config['model'])
    logger.info('OK')

    reporter = None
    if ReporterClass:
        logger.info('Initializing reporter....')
        reporter = ReporterClass(config, logger)
        logger.info('OK')

    logger.info('Initializing inferencer....')
    if reporter:
        inferencer = InferencerClass(model, logger, config, reporter)
    else:
        inferencer = InferencerClass(model, logger, config)
    logger.info('OK')

    logger.info('Initiating inferencer run...')
    inferencer.run(test_set)
    logger.info('inferencer run COMPLETED')

    if reporter:
        reporter.wandb_finish()

if __name__ == '__main__':
    main()
