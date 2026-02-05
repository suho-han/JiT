#!/bin/bash
set -e

# OCTA segmentation training
# Image-conditioned mask diffusion: predict mask (1ch) conditioned on image (1ch)

# GPU selection (default: 0,1)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Training arguments:
# --model: Model architecture (JiT-B/16 = Base model with 16x16 patches)
# --proj_dropout: Dropout rate for projection layer
# --P_mean, --P_std: Mean and std for noise schedule distribution
# --img_size: Input image size (256x256)
# --noise_scale: Scaling factor for noise
# --batch_size: Training batch size per GPU
# --blr: Base learning rate
# --epochs: Total training epochs
# --warmup_epochs: Number of warmup epochs for learning rate
# --output_dir: Directory to save checkpoints
# --resume: Directory to resume training from
# --data_path: Path to training dataset
# --img_channel: Number of input image channels (1 for grayscale)
# --mask_channel: Number of output mask channels (1 for binary segmentation)
# --online_eval: Enable validation during training
# --eval_freq: Frequency (in epochs) for evaluation
# --val_use_patch: Use patch-based inference during validation (matches inference_jit.py)
# --val_patch_size: Patch size for validation (256x256)
# --val_stride: Stride for patch extraction (128 = 50% overlap)
# --val_ema: EMA model to use for validation (ema1 provides better generalization)
# --val_threshold: Threshold for binarization in metrics (0.0 for [-1,1] range)

uv run torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=29501 \
src/main_jit.py \
--model JiT-H/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 4 --blr 5e-5 \
--epochs 10000 --warmup_epochs 100 \
--data_path data/OCTA500_6M/ --dataset OCTA500_6M \
--img_channel 1 --mask_channel 1 \
--online_eval

uv run python alert.py --repo jit --message "OCTA segmentation 10,000 epochs training completed"
# GPU selection (default: 0,1)      
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Testing arguments:
# --checkpoint: Path to the trained checkpoint
# --data_path: Path to testing dataset
# --output_dir: Directory to save inference results
# --batch_size: Inference batch size
# --num_workers: Number of data loading workers
# --img_size: Input image size (256x256)
# --img_channel: Number of input image channels (1 for grayscale)
# --mask_channel: Number of output mask channels (1 for binary segmentation)

uv run python src/inference_jit.py --model JiT-H/32 --checkpoint checkpoint-last.pth \
--data_path data/OCTA500_6M/test --output_dir outputs \
--batch_size 4 --num_workers 0 --img_size 256 --img_channel 1 --mask_channel 1 \
--samp_patch_size 256 --stride 128 --metrics --device cuda

uv run python alert.py --repo jit --message "OCTA segmentation inference completed"