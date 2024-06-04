#! /bin/bash
#SBATCH --job-name='test'
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0-0:10
#SBATCH --output='test.out'


# Executable
python3 test.py --config-path='./' --config-name='config_tss.yaml' --mode='tss_spe'