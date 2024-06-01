#! /bin/bash
#SBATCH --job-name="train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0-0:10
#SBATCH --output="train.out"


# Executable
python3 train.py --config-path="./" --config-name="config_tss.yaml" --mode='tss_spe'