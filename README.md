# NYCU Computer Vision 2026 HW4

- Student ID: 314554036
- Name: 郭彥頡, Yenchieh Kuo

## Introduction
The objective of this assignment is to train a single model from scratch capable of restoring images degraded by two different conditions: rain and snow. The core challenge lies in the "All-in-One" nature of the task and the hardware constraint of training from scratch on an 8GB VRAM GPU (RTX 4060).

To tackle this, the method builds upon the fundamental design of **PromptIR** [1], which utilizes a Prompt Generation Module (PGM) to implicitly deduce the degradation type (rain or snow) and a Prompt Interaction Module (PIM) to guide the restoration dynamically. However, since the original Transformer-based Restormer blocks in PromptIR consume excessive memory, I designed a **CNN-Transformer Hybrid Architecture**. By integrating memory-efficient **NAFBlocks** [2] with Transformer bottlenecks, alongside Progressive Fine-tuning, CutBlur augmentation, and specialized loss functions, the model successfully achieved a highly competitive PSNR of **28.14** dB under strict hardware constraints.

## Environment Setup
It is recommended to use Miniconda to set up the environment. You can easily recreate the environment using the provided ".yml" file:

```bash
conda env create -f vrdl_hw4.yml
conda activate vrdl_hw4
```

## Usage
### Training
How to train your model. The following three .py should be execute sequentially.

Make sure your .pth file is on the right direction.
```bash
python train.py
python train_FT_v1.py
python train_FT_v2.py
```
### Inference
How to run inference. Same as training, you need to execute them sequentially.
```bash
python inference.py
python inference_FT_v1.py
python inference_FT_v2.py
```

## Performance Snapshot
![Leaderboard](CodeBench_competition.png)
