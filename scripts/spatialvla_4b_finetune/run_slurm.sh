#!/bin/bash
#SBATCH --job-name=spatialvla_finetune
#SBATCH --account=MST113264
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules and activate conda
module load gcc/11.5.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatialvla

# Create logs directory if it doesn't exist
mkdir -p logs

# Set training parameters
export DEBUG=false
export GPUS=1
export GPUS_PER_NODE=1
export PER_DEVICE_BATCH_SIZE=16  # Total batch size = 1 * 16 = 16
export epoch=1                   # Run for 1 epoch
export save_steps=10000          # Save checkpoint every 10000 steps

# Run the training script
bash scripts/spatialvla_4b_finetune/finetune_lora.sh
