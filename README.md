# prototype_based_latent_diffusion

This repository contains code for training a prototype-based latent diffusion model using PyTorch. The model includes a UNet for image generation and an Autoencoder for latent space encoding and decoding. It supports training from scratch and resuming from checkpoints, with optional use of Distributed Data Parallel (DDP) for multi-GPU training.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Resuming Training](#resuming-training)
- [File Structure](#file-structure)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Diffusers library
- TQDM
- NumPy
- Pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/swapnilbhat/prototype_based_latent_diffusion.git
   cd prototype_based_latent_diffusion

2. Install the required packages:
    ```bash
   pip install torch torchvision diffusers accelerate tqdm pandas
    ```

3. Download the dataset and extract it in the root directory:

   https://www.kaggle.com/datasets/andrewmvd/bone-marrow-cell-classification

4. Training
   - To train the model from scratch, use the following command:

     ```bash
     torchrun --nproc_per_node=NUM_GPUS train.py
     ```

   - To resume training from a checkpoint, specify the `--path_resume` argument with the path to the checkpoint directory:

     ```bash
     torchrun --nproc_per_node=NUM_GPUS train.py --path_resume PATH_TO_CHECKPOINT
     ```

5. File Structure
- train.py: Main script for training and resuming the latent diffusion model.
- unet.py: Contains the UNet model definition.
- diffusion.py: Contains the Gaussian Diffusion model.
- utils.py: Utility functions.
- embedding.py: Contains the Conditional Embedding model.
- scheduler.py: Learning rate scheduler.
- dataloaders.py: Data loading utilities.
- sample.py: Main script for sampling and image generation

