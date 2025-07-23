#!/bin/bash
#SBATCH --job-name=extract_embeddings
#SBATCH --output=/private/home/francoisporcher/dinov2/francois/logs/extract_embeddings_%j.log
#SBATCH --error=/private/home/francoisporcher/dinov2/francois/logs/extract_embeddings_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8                  # Number of GPUs you want (adjust to your node)
#SBATCH --time=12:00:00           # Max time
#SBATCH --mem=64G                 # RAM requested
#SBATCH --partition=learnlab
#SBATCH --gpus-per-node=8           # 8 GPUs on that node
#SBATCH --constraint=ampere80gb
#SBATCH --partition=learnlab

# module load anaconda3/2023.03
# module load cuda/12.2

# Activate your environment
source ~/.bashrc
conda activate dinov2  # or whatever env you use

# Move to the script directory
cd /private/home/francoisporcher/dinov2/francois/scripts

# torchrun launches your script with DDP over N GPUs
torchrun --nproc_per_node=8 extract_imagenet_embeddings.py \
    --image_resolution 70 \
    --batch_size 256 \
    --output_dir /private/home/francoisporcher/imagenet_embeddings