#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3
# uv run python src/inference_jit.py --model JiT-B/16 --checkpoint checkpoint-9000.pth \
# --data_path ./data/OCTA500_6M/test --output_dir outputs \
# --batch_size 4 --num_workers 0 --img_size 256 --img_channel 1 --mask_channel 1 \
# --samp_patch_size 256 --stride 128 --metrics --device cuda

uv run python src/inference_jit.py --model JiT-H/32 --checkpoint checkpoint-7000.pth \
--data_path ./data/OCTA500_6M/test --output_dir outputs \
--batch_size 4 --num_workers 0 --img_size 256 --img_channel 1 --mask_channel 1 \
--samp_patch_size 256 --stride 128 --metrics --device cuda

uv run alert.py --repo jit --message "OCTA segmentation inference completed"