
Fine-Tuning EfficientNetV2-S on iNaturalist12K
=============================================

This repository contains a PyTorch-based script for fine-tuning the EfficientNetV2-S architecture on a custom image classification dataset (iNaturalist12K subset of nature_12K). It follows a 3-stage gradual unfreezing strategy and logs all training metrics to Weights & Biases (wandb).

Dataset Structure
-----------------

Make sure your dataset is organized as follows:

nature_12K/
└── inaturalist_12K/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── val/
        ├── class_1/
        ├── class_2/
        └── ...

Getting Started
---------------

1. Install Dependencies

    pip install torch torchvision pytorch-lightning wandb

2. Initialize wandb

    wandb login

Training Logic
--------------

The training is performed in 3 stages:

1. Stage 1: Train only the classifier and freeze other layers 
2. Stage 2: Unfreeze the last feature convolutional and the classifier.
3. Stage 3: Unfreeze the entire model and fine-tune.

Configuration
-------------

All key training hyperparameters are logged and managed through wandb.config:

    wandb.init(project="iNaturalist_EffNetV2S_finetune_3", config={
        "architecture": "EfficientNetV2-S",
        "dataset": "iNaturalist12K",
        "num_classes": 10,
        "batch_size": 32,
        "epochs_stage1": 30,
        "epochs_stage2": 30,
        "epochs_stage3": 35,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-4,
        "lr_stage3": 1e-5,
        "img_size": 224,
    })

Functional Overview
-------------------

train()
    Main function that:
    - Initializes wandb
    - Prepares data transforms and dataloaders
    - Loads a pretrained EfficientNetV2-S model
    - Applies 3-stage fine-tuning (classifier → last block → full model)
    - Logs performance to wandb

run_epoch(stage)
    Helper function to:
    - Train and evaluate the model for one epoch
    - Compute and log loss/accuracy metrics

Data Augmentation
-----------------

Augmentations used for training include:
- Random resized crop
- Horizontal flip
- Affine transformations
- Color jittering

Validation data is center-cropped and normalized using standard ImageNet statistics.

Logging & Monitoring
--------------------

All training metrics are logged to your wandb project in real time, including:
- Train/val loss and accuracy per stage
- Learning rate used
- Configuration summary

Paths to Edit
-------------

Update the dataset paths in train() before running:

    train_dir = "/home/user/kartikey_phd/DA6401/nature_12K/inaturalist_12K/train"
    val_dir   = "/home/user/kartikey_phd/DA6401/nature_12K/inaturalist_12K/val"

Running the Script
------------------

    python PartB.py

Make sure to replace your_script_name.py with the actual filename.

Notes
-----

- This script is optimized for GPU training (with auto CUDA detection).
- Ensure your dataset has exactly num_classes defined in the config.
- You can easily adapt the script for a different model by modifying the model section.
