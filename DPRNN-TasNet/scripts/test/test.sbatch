#! /bin/bash
#SBATCH --job-name="dprnntasnet_inferencer"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0-24:0
#SBATCH --output="test.out"


# Executable
python3 test.py --config-path="./" --config-name="config.yaml"