#!/bin/bash
set -e

# OCTA segmentation inference refactored
# Usage: uv run bash scripts/test.sh <dataset> <model> <step> <device> [num_samples] [add_loss] [cond_weight] [soft_vote] [batch_size]

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: uv run bash scripts/test.sh <dataset> <model> <step> <device> [num_samples] [add_loss] [cond_weight] [soft_vote] [batch_size]"
    echo "Example: uv run bash scripts/test.sh OCTA500_6M JiT-B/16 10000 0 5 weighted_dice_bce fff True 16"
    exit 0
fi

# Load shared utilities
source scripts/utils.sh

DATASET=${1:-OCTA500_6M}
MODEL=${2:-JiT-B/16}
STEP=${3:-last}
DEVICE=${4:-0}
NUM_SAMPLES=${5:-1}
ADD_LOSS=${6:-""}
COND_WEIGHT=${7:-""}
SOFT=${8:-False}
BS=${9:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$DEVICE}

# Get model and dataset configs
MODEL_CFG=($(get_model_config "$MODEL")) || { echo "Unknown model: $MODEL"; exit 1; }
IMG_SIZE=${MODEL_CFG[0]}
SAMP_PATCH_SIZE=${MODEL_CFG[1]}

DATA_CFG=($(get_dataset_config "$DATASET")) || { echo "Unknown dataset: $DATASET"; exit 1; }
DATA_PATH=${DATA_CFG[0]}
IMG_CHANNEL=${DATA_CFG[1]}
MASK_CHANNEL=${DATA_CFG[2]}

CHECKPOINT="checkpoint-${STEP}.pth"
TEST_PATH="${DATA_PATH}test"

# Argument processing
if [ "$SOFT" = "True" ] || [ "$SOFT" = "true" ]; then SOFT_ARGS=(--soft_vote); else SOFT_ARGS=(); fi
ADD_LOSS_ARGS=($(get_loss_args "$ADD_LOSS"))
PARSED_CW=$(parse_cond_weight "$COND_WEIGHT")
if [ -n "$PARSED_CW" ]; then COND_WEIGHT_ARGS=(--cond_weight "$PARSED_CW"); fi

echo "Starting inference: $DATASET / $MODEL / step $STEP / ensemble $NUM_SAMPLES"
echo "Add loss: ${ADD_LOSS_ARGS[*]}"
echo "Cond weight: ${COND_WEIGHT_ARGS[*]}"
echo "Online eval: $ONLINE_EVAL"

uv run python src/inference_jit.py \
    --model "$MODEL" --checkpoint "$CHECKPOINT" \
    --data_path "$TEST_PATH" --dataset "$DATASET" --output_dir outputs \
    --batch_size "$BS" --num_workers 0 --img_size "$IMG_SIZE" \
    --img_channel "$IMG_CHANNEL" --mask_channel "$MASK_CHANNEL" \
    --samp_patch_size "$SAMP_PATCH_SIZE" --stride "$SAMP_PATCH_SIZE" \
    --num_samples "$NUM_SAMPLES" --metrics "${SOFT_ARGS[@]}" \
    "${ADD_LOSS_ARGS[@]}" "${COND_WEIGHT_ARGS[@]}"

uv run python scripts/alert.py --repo jit --message "${DATASET} segmentation inference completed ${MODEL} @ epoch ${STEP}"
