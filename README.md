# tss-with-dprnn
This repository is a part of my bachelor thesis: "Target Speech Separation with Dual-Path RNN."

## Description

This repository contains pre-trained models for Target Speech Separation (TSS) based on Dual-Path RNN. Below are brief descriptions of the models:

- **DPRNN-TasNet**: This model is for blind speech separation (BSS), proposed by [Luo et al. (2020)](https://arxiv.org/abs/1910.06379). The implementation is based on [Asteroid's](https://github.com/asteroid-team/asteroid) version.

- **DPRNN-Spe**: This model is based on DPRNN-TasNet and SpEx+. Its implementation is somewhat similar to the model described in [Deng et al. (2022)](https://arxiv.org/abs/2011.02102), hence the naming. However, a pre-trained DPRNN-TasNet was adapted for BSS (2 voices) rather than training the whole DPRNN-Spe from scratch. Additionally, different fusion approaches were tested: addition, [attention](https://arxiv.org/pdf/2010.10923), concatenation, multiplication, and FiLM (multiplication + addition). 

- **DPRNN-Spe-IRA**: An adaptation of DPRNN-Spe using the Iterative Refined Adaptation (IRA) strategy, as proposed by [Deng et al. (2022)](https://arxiv.org/abs/2011.02102).

- **DPRNN-RawNet**: This model combines DPRNN-TasNet with [RawNet3](https://arxiv.org/pdf/2203.08488).

All configuration parameters are available in the `scripts` folder. Pre-trained models are saved in the `chkpts` folder.

## Project UML
The library scheme is available in TODO.

## Datasets

The Libri2Mix dataset (`train-100`, `dev`, `test`) was used for training, validation, and testing. A randomly selected audio was used as a reference in the TSS task, which was different from the target one but contained the voice of the same speaker. The dataset is available on Kaggle, and the `.pkl` file can be found in the `datasets` folder.

All audio recordings have a sample rate of 8 kHz. The dataset was generated in `min` mode.

## Running
### Installation
Run the following command to install all required libraries:
```
pip install -r requirements.txt
```
Additionally, Libri2Mix installation is required. 

### Training
The training script is located in the `scripts/train` folder. Refer to the configuration files for details. Example usage:
```
python3 train.py --config-path="./" --config-name="config_tss.yaml" --mode='tss_spe'
```

### Testing
The testing script is located in the `scripts/test` folder. Refer to the configuration files for details. Example usage:
```
python3 test.py --config-path='./' --config-name='config_tss.yaml' --mode='tss_spe'
```
 
### MRE
Notebook `example.ipynb` contains a minimal reproducible example. As MiniLibriMix will be used, there is no need for Libri2Mix installation.