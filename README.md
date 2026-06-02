# NYCU Computer Vision 2026 HW4

- Student ID: 314554036
- Name: 郭彥頡, Yenchieh Kuo

## Introduction
This project implements an image classification model using Deep Learning. 
I utilized multiple ResNet models like **ResNet34, RseNet50, ResNet101 and ResNet152** as the backbone and applied Transfer Learning with ImageNet pre-trained weights.
To improve the performance and avoid overfitting, 3 learning rate setting were implement, warm-up -> Hold -> Decay, respectivily.

## Environment Setup
It is recommended to use Miniconda to set up the environment. You can easily recreate the environment using the provided ".yml" file:

```bash
# Create the environment from the environment.yml file
conda env create -f VRDL_HW1_env.yml

# Activate the new environment
conda activate VRDL_HW1_env
```

## Usage
### Training
How to train your model.
```bash
python VRDL_HW1_ResNet152.py
```
If you want to use other ResNet model, choose other .py file you want.
### Inference
How to run inference.
```bash
#Make sure the dataset is in ./data/test
python VRDL_HW1_ResNet101.py --mode test --weights ./best_model_resnet101.pth --data_dir ./data/test
```

## Performance Snapshot
![Leaderboard](CodeBench_competition.png)
```
