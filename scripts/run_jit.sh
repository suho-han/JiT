#!/bin/bash
set -e

# OCTA segmentation training
# Image-conditioned mask diffusion: predict mask (1ch) conditioned on image (1ch)


# Usage:
#   uv run bash scripts/run_jit.sh <dataset> <model> <device>
# Example:
#   uv run bash scripts/run_jit.sh OCTA500_6M JiT-B/16 3

DATASET=${1:-OCTA500_6M}
MODEL=${2:-JiT-B/16}
DEVICE=${3:-3}
EPOCH=${4:-""}
# GPU selection (default: 3)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEVICE}


case "$MODEL" in
    */16)
        IMG_SIZE=256
		SAMP_PATCH_SIZE=256
    ;;
    */32)
        IMG_SIZE=512
        SAMP_PATCH_SIZE=512
    ;;
esac

if [ -z "$EPOCH" ]; then
    case "$MODEL" in
        JiT-B/*)
            EPOCH=10000
            WARMUP_EPOCH=10
            ;;
        JiT-L/*)
            EPOCH=20000
            WARMUP_EPOCH=20
            ;;
        JiT-H/*)
            EPOCH=40000
            WARMUP_EPOCH=40
            ;;
        JiT_CondImg-B/*)
            EPOCH=20000
            WARMUP_EPOCH=20
            ;;
        JiT_CondImg-L/*)
            EPOCH=40000
            WARMUP_EPOCH=40
            ;;
        JiT_CondImg-H/*)
            EPOCH=80000
            WARMUP_EPOCH=80
            ;;
        *)
            echo "Unknown model: $MODEL"
            echo "Supported: JiT-B/L/H, JiT_CondImg-B/L/H"
            exit 1
    esac
else
    WARMUP_EPOCH=$((EPOCH / 1000))
fi

case "$DATASET" in
	OCTA500_6M)
		DATA_PATH="data/OCTA500_6M/"
		IMG_CHANNEL=1
		MASK_CHANNEL=1
        BATCH_SIZE=32
        ONLINE_EVAL="--online_eval"
		;;
	MoNuSeg)
		DATA_PATH="data/MoNuSeg/"
		IMG_CHANNEL=3
		MASK_CHANNEL=1
        BATCH_SIZE=16
        ONLINE_EVAL=""
		;;
	ISIC2016)
		DATA_PATH="data/ISIC2016/"
		IMG_CHANNEL=3
		MASK_CHANNEL=1
        BATCH_SIZE=32
        ONLINE_EVAL=""
		;;
	ISIC2018)
		DATA_PATH="data/ISIC2018/"
		IMG_CHANNEL=3
		MASK_CHANNEL=1
        BATCH_SIZE=128
        ONLINE_EVAL="--online_eval"
		;;
	*)
		echo "Unknown dataset: $DATASET"
		echo "Supported: OCTA500_6M, MoNuSeg, ISIC2016/2018"
		exit 1
		;;
esac

# Training arguments:
# OCTA500_6M
# data_path data/OCTA500_6M/ --dataset OCTA500_6M --batch_size 32 --img_channel 1 --mask_channel 1 --img_size 256 --online_eval
# --samp_patch_size 256 --stride 128
# MoNuSeg
# data_path data/MoNuSeg/ --dataset MoNuSeg --batch_size 16 --img_channel 3 --mask_channel 1 --img_size 256
# --samp_patch_size 256 --stride 128
# ISIC
# data_path data/ISIC2018/ --dataset ISIC2018 --batch_size 16 --img_channel 3 --mask_channel 1 --img_size 256
# --samp_patch_size 256 --stride 128

FIRST_DEVICE=${DEVICE%%,*}

echo src/main_jit.py \
--model "$MODEL" \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 --noise_scale 1.0 \
--batch_size "$BATCH_SIZE" --blr 5e-5 \
--epochs "$EPOCH" --warmup_epochs "$WARMUP_EPOCH" \
--data_path "$DATA_PATH" --dataset "$DATASET" \
--img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" --img_size "$IMG_SIZE" \
"$ONLINE_EVAL"

TRAIN_ARGS=(
	src/main_jit.py
	--model "$MODEL"
	--proj_dropout 0.0
	--P_mean -0.8 --P_std 0.8 --noise_scale 1.0
	--batch_size "$BATCH_SIZE" --blr 5e-5
	--epochs "$EPOCH" --warmup_epochs "$WARMUP_EPOCH" \
	--data_path "$DATA_PATH" --dataset "$DATASET" \
	--img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" --img_size "$IMG_SIZE"
)

if [[ -n "$ONLINE_EVAL" ]]; then
	TRAIN_ARGS+=("$ONLINE_EVAL")
fi

uv run python \
"${TRAIN_ARGS[@]}"

uv run python scripts/alert.py --repo jit --message "${DATASET} / ${MODEL} ${EPOCH} epochs training completed"
# GPU selection first device (default: 3)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$FIRST_DEVICE}

# Testing arguments:
# --checkpoint: Path to the trained checkpoint
# --data_path: Path to testing dataset
# --output_dir: Directory to save inference results
# --batch_size: Inference batch size
# --num_workers: Number of data loading workers
# --img_size: Input image size (256x256)
# --img_channel: Number of input image channels (1 for grayscale)
# --mask_channel: Number of output mask channels (1 for binary segmentation)

uv run python src/inference_jit.py --model "$MODEL" --checkpoint "checkpoint-${EPOCH}.pth" \
--data_path "${DATA_PATH}test" --output_dir outputs \
--batch_size 4 --num_workers 0 --img_size "$IMG_SIZE" --img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" \
--metrics --device cuda \
--samp_patch_size "${SAMP_PATCH_SIZE}" --stride 128

uv run python scripts/alert.py --repo jit --message "${DATASET} / ${MODEL} inference completed"