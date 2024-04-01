import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append('../../')

@hydra.main(version_base=None, config_path='./', config_name='dprnntasnet_config')
def main(config: DictConfig):
    OmegaConf.resolve(config)

    print(config['name'])

    print("Initializing Model and putting to device.....")
    model = hydra.utils.instantiate(config["model"])
    model.eval()
    
if __name__ == '__main__':   
    main()