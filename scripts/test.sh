#!/bin/bash
set -e

# Usage:
#   uv run bash scripts/test.sh <dataset> <model> <step> <device>
# Example:
#   uv run bash scripts/test.sh OCTA500_6M JiT-B/16 last 0

DATASET=${1:-OCTA500_6M}
MODEL=${2:-JiT-B/16}
STEP=${3:-last}
DEVICE=${4:-0}
SOFT=${5:-False}

# GPU selection (default: 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEVICE}

case "$DATASET" in
	OCTA500_6M)
		DATA_PATH="data/OCTA500_6M/"
		IMG_CHANNEL=1
		MASK_CHANNEL=1
		IMG_SIZE=256
		PATCH_SIZE=256
		;;
	MoNuSeg)
		DATA_PATH="data/MoNuSeg/"
		IMG_CHANNEL=3
		MASK_CHANNEL=1
		IMG_SIZE=256
		PATCH_SIZE=256
		;;
	ISIC2016)
		DATA_PATH="data/ISIC2016/"
		IMG_CHANNEL=3
		MASK_CHANNEL=1
		IMG_SIZE=256
		PATCH_SIZE=256
		;;
    ISIC2018)
        DATA_PATH="data/ISIC2018/"
        IMG_CHANNEL=3
        MASK_CHANNEL=1
        IMG_SIZE=256
        PATCH_SIZE=256
        ;;
	*)
		echo "Unknown dataset: $DATASET"
		echo "Supported: OCTA500_6M, MoNuSeg, ISIC2016, ISIC2018"
		exit 1
		;;
esac

CHECKPOINT="checkpoint-${STEP}.pth"
TEST_PATH="${DATA_PATH}test"
# OCTA500_6M
# --data_path data/OCTA500_6M/ --dataset OCTA500_6M --batch_size 32 --img_channel 1 --mask_channel 1 --img_size 256 --online_eval
# --samp_patch_size 256 
# MoNuSeg
# --data_path data/MoNuSeg/ --dataset MoNuSeg --batch_size 16 --img_channel 3 --mask_channel 1 --img_size 512
# --samp_patch_size 512
# ISIC2016
# --data_path data/ISIC2016/ --dataset ISIC2016 --batch_size 16 --img_channel 3 --mask_channel 1 --img_size 512
# --samp_patch_size 512
# ISIC2018
# --data_path data/ISIC2018/ --dataset ISIC2018 --batch_size 16 --img_channel 3 --mask_channel 1 --img_size 512
# --samp_patch_size 512

if [ "$SOFT" = "True" ] || [ "$SOFT" = "true" ]; then
    SOFT="--soft_vote"
else
    SOFT=""
fi

echo --model "$MODEL" --checkpoint "$CHECKPOINT" \
--data_path "$TEST_PATH" --dataset "$DATASET" --output_dir outputs \
--batch_size 32 --num_workers 0 --img_size "$IMG_SIZE" --img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" \
--samp_patch_size "$PATCH_SIZE" --stride "$PATCH_SIZE" --metrics $SOFT

uv run python src/inference_jit.py --model "$MODEL" --checkpoint "$CHECKPOINT" \
--data_path "$TEST_PATH" --dataset "$DATASET" --output_dir outputs \
--batch_size 32 --num_workers 0 --img_size "$IMG_SIZE" --img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" \
--samp_patch_size "$PATCH_SIZE" --stride "$PATCH_SIZE" --metrics $SOFT


uv run scripts/alert.py --repo jit --message "${DATASET} segmentation inference completed ${MODEL} @ epoch ${STEP}"