#!/bin/bash
set -e

# OCTA segmentation training refactored
# Usage: uv run bash scripts/run_jit.sh <dataset> <model> <device> [add_loss] [cond_weight] [epoch]

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: uv run bash scripts/run_jit.sh <dataset> <model> <device> [add_loss] [cond_weight] [epoch]"
    echo ""
    echo "Example: uv run bash scripts/run_jit.sh OCTA500_6M JiT-B/16 3 weighted_dice_bce fff"
    exit 0
fi

# Load shared utilities
source scripts/utils.sh

DATASET=${1:-OCTA500_6M}
MODEL=${2:-JiT-B/16}
DEVICE=${3:-0}
ADD_LOSS=${4:-""}
COND_WEIGHT=${5:-""}
EPOCH=${6:-""}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEVICE}

# Get model and dataset configs
MODEL_CFG=($(get_model_config "$MODEL")) || { echo "Unknown model: $MODEL"; exit 1; }
IMG_SIZE=${MODEL_CFG[0]}
SAMP_PATCH_SIZE=${MODEL_CFG[1]}

DATA_CFG=($(get_dataset_config "$DATASET")) || { echo "Unknown dataset: $DATASET"; exit 1; }
DATA_PATH=${DATA_CFG[0]}
IMG_CHANNEL=${DATA_CFG[1]}
MASK_CHANNEL=${DATA_CFG[2]}
BATCH_SIZE=${DATA_CFG[3]}
ONLINE_EVAL=${DATA_CFG[4]}

# Epoch logic
if [ -z "$EPOCH" ]; then
    case "$MODEL" in
        JiT-B/*) EPOCH=10000 ;;
        JiT-L/*) EPOCH=20000 ;;
        JiT-H/*) EPOCH=40000 ;;
        JiT_CondImg-B/* | JiT_ParaCond-B/* | JiT_ParaCondWave-B/*) EPOCH=30000 ;;
        JiT_CondImg-L/* | JiT_ParaCond-L/* | JiT_ParaCondWave-L/*) EPOCH=50000 ;;
        JiT_CondImg-H/* | JiT_ParaCond-H/* | JiT_ParaCondWave-H/*) EPOCH=80000 ;;
        *) echo "Unknown model: $MODEL"; exit 1 ;;
    esac
fi
WARMUP_EPOCH=$((EPOCH / 1000))

# Argument processing
ADD_LOSS_ARGS=($(get_loss_args "$ADD_LOSS"))
PARSED_CW=$(parse_cond_weight "$COND_WEIGHT")
if [ -n "$PARSED_CW" ]; then COND_WEIGHT_ARGS=(--cond_weight "$PARSED_CW"); fi

NUM_GPUS=$(echo "$DEVICE" | tr ',' '\n' | wc -l)

echo "Starting training: $DATASET / $MODEL / $EPOCH epochs / GPUs $DEVICE"
echo "Add loss: ${ADD_LOSS_ARGS[*]}"
echo "Cond weight: ${COND_WEIGHT_ARGS[*]}"
echo "Online eval: $ONLINE_EVAL"

MASTER_PORT=$((29000 + RANDOM % 1000))
uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    src/main_jit.py \
    --model "$MODEL" \
    --proj_dropout 0.0 \
    --P_mean -0.8 --P_std 0.8 --noise_scale 1.0 \
    --batch_size "$BATCH_SIZE" --blr 5e-5 \
    --epochs "$EPOCH" --warmup_epochs "$WARMUP_EPOCH" \
    --data_path "$DATA_PATH" --dataset "$DATASET" \
    --img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" --img_size "$IMG_SIZE" \
    $ONLINE_EVAL "${ADD_LOSS_ARGS[@]}" "${COND_WEIGHT_ARGS[@]}"

uv run python scripts/alert.py --repo jit --message "${DATASET} / ${MODEL} ${EPOCH} epochs training completed"

# Final testing
FIRST_DEVICE=${DEVICE%%,*}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$FIRST_DEVICE}

echo "Running final inference..."
uv run python src/inference_jit.py --model "$MODEL" --checkpoint "checkpoint-${EPOCH}.pth" \
    --dataset "$DATASET" --data_path "${DATA_PATH}test" --output_dir outputs \
    --batch_size 4 --num_workers 0 --img_size "$IMG_SIZE" --img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" \
    --metrics --device cuda --samp_patch_size "${SAMP_PATCH_SIZE}" --stride 128 \
    "${ADD_LOSS_ARGS[@]}" "${COND_WEIGHT_ARGS[@]}"

uv run python scripts/alert.py --repo jit --message "${DATASET} / ${MODEL} inference completed"
