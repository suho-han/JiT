#!/bin/bash

# Shared utilities for JiT scripts

get_model_config() {
    local model=$1
    case "$model" in
        */16)
            echo "256 256" # IMG_SIZE SAMP_PATCH_SIZE
            ;;
        */32)
            echo "512 512"
            ;;
        *)
            return 1
            ;;
    esac
}

get_dataset_config() {
    local dataset=$1
    case "$dataset" in
        OCTA500_6M)
            echo "data/OCTA500_6M/ 1 1 8 --online_eval" # PATH IMG_CH MASK_CH BS EXTRA
            ;;
        MoNuSeg)
            echo "data/MoNuSeg/ 3 1 16"
            ;;
        ISIC2016)
            echo "data/ISIC2016/ 3 1 32"
            ;;
        ISIC2018)
            echo "data/ISIC2018/ 3 1 128 --online_eval"
            ;;
        *)
            return 1
            ;;
    esac
}

parse_cond_weight() {
    local cond_weight=$1
    if [[ ${#cond_weight} -eq 3 && "$cond_weight" =~ ^[fslz]+$ ]]; then
        get_cond_type() {
            case $1 in
                f) echo "fixed" ;;
                s) echo "shared" ;;
                l) echo "learnable" ;;
                z) echo "learnable_0" ;;
            esac
        }
        local cw1=$(get_cond_type "${cond_weight:0:1}")
        local cw2=$(get_cond_type "${cond_weight:1:1}")
        local cw3=$(get_cond_type "${cond_weight:2:1}")
        echo "{'cond': '$cw1', 'low_cond': '$cw2', 'high_cond': '$cw3'}"
    else
        echo "$cond_weight"
    fi
}

get_loss_args() {
    local add_loss=$1
    if [ -n "$add_loss" ]; then
        echo "--add_loss --add_loss_name $add_loss --add_loss_weight 1.0"
    fi
}
