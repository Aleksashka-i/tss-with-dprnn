#! /bin/bash
#SBATCH --job-name="dprnnspetasnet_trainer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=0-24:0
#SBATCH --output="train_spe.out"


# Executable
python3 train_spe.py --config-path="./" --config-name="config.yaml"