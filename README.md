# Towards accurate and reliable ICU outcome prediction: a multimodal learning framework based on belief function theory using structured EHRs and free-text notes

## Overview

This repository provides a demo code for an BFT-based multimodal fusion approach on EHRs for ICU outcome prediction.

## Installation

### Clone the repository

```
git clone https://github.com/yuchengruan/evid_multimodal_ehr.git
cd evid_multimodal_ehr
```

### Create a conda environment and install dependencies (optional but recommended)

```
conda env create -f environment.yml
conda activate py_38
```

## Data acquisition

In this study, we utilized two multimodal EHR datasets: 

- MIMIC-III: https://physionet.org/content/mimiciii/1.4/
- ZICIP: https://physionet.org/content/icu-infection-zigong-fourth/1.1/

Due to the requirements of data provider, you have to the complete the training program [CITI Data or Specimens Only Research](https://physionet.org/content/icu-infection-zigong-fourth/view-required-training/1.1/#1) and sign the data use requirement before accessing the data.

## Data preprocessing

Under data directory `./data/datasets/mimic/`, run the following python script for data preprocessing:

- process_step_1.py: preprocess the data for our proposed method and baselines
- process_step_2.py: extract the embeddings from free-text notes

## Model training

In this project, we utilize [Comet](https://www.comet.com/site/?utm_source=chatgpt.com) for experiment management. To connect to the service, you’ll need to configure your login credentials in [utils/logger.py](./utils/logger.py).

We provide a Jupyter notebook [train_enn.ipynb](./train_enn.ipynb) that demonstrates the training and validation processes. You can manually configure the hyperparameters by adjusting the `hparams_dict` in the notebook.

For a seamless experience, we recommend running the experiment via [train_enn.py](./train_enn.py). This script allows you to easily manage experiments with various hyperparameter configurations. To train with the default hyperparameters, simply run:

```
python train_enn.py
```

