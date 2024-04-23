#! /bin/bash
#SBATCH --job-name="dprnntasnet_trainer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=0-24:0
#SBATCH --output="train.out"


# Executable
python3 train.py --config-path="./" --config-name="config.yaml"