#!/bin/bash

# List of directories
dirs=(
"outputs/JiT_CondImg-B-16-ISIC2016"
"outputs/JiT_CondImg-B-16-MoNuSeg"
"outputs/JiT_CondImg-B-16-OCTA500_6M"
"outputs/JiT_CondImg-H-16-OCTA500_6M"
"outputs/JiT_CondImg-L-16-OCTA500_6M"
"outputs/JiT_ParaCond-B-16-ISIC2016"
"outputs/JiT_ParaCond-B-16-MoNuSeg"
"outputs/JiT_ParaCond-B-16-OCTA500_6M"
"outputs/JiT_ParaCond-H-16-MoNuSeg"
"outputs/JiT_ParaCond-H-16-OCTA500_6M"
"outputs/JiT_ParaCond-L-16-OCTA500_6M"
"outputs/JiT_ParaCondWave-B-16-ISIC2016"
"outputs/JiT_ParaCondWave-B-16-MoNuSeg"
"outputs/JiT_ParaCondWave-B-16-OCTA500_6M"
"outputs/JiT_ParaCondWave-H-16-MoNuSeg"
"outputs/JiT_ParaCondWave-H-16-OCTA500_6M"
"outputs/JiT_ParaCondWave-L-16-OCTA500_6M"
"outputs/JiT-B-16-ISIC2016"
"outputs/JiT-B-16-MoNuSeg"
"outputs/JiT-B-16-OCTA500_6M"
"outputs/JiT-B-32-OCTA500_6M"
"outputs/JiT-H-16-MoNuSeg"
"outputs/JiT-H-16-OCTA500_6M"
"outputs/JiT-L-16-OCTA500_6M"
"outputs/JiT-L-32-OCTA500_6M"
)

DRY_RUN=1

if [ "$1" == "--run" ]; then
    DRY_RUN=0
fi

for dir in "${dirs[@]}"; do
    ckpt_dir="${dir}/checkpoints"
    if [ -d "$ckpt_dir" ]; then
        # loop through files in checkpoints directory
        for file in "$ckpt_dir"/checkpoint-*.pth; do
            # Check if file exists to handle empty case
            [ -e "$file" ] || continue
            
            # Extract the number from the filename
            filename=$(basename "$file")
            # Usually checkpoint-1000.pth
            step="${filename%.pth}"      # checkpoint-1000
            step="${step#checkpoint-}" # 1000
            
            # Check if step is a valid integer
            if [[ "$step" =~ ^[0-9]+$ ]]; then
                if [ $((step % 2000)) -ne 0 ]; then
                    if [ $DRY_RUN -eq 1 ]; then
                        echo "[DRY RUN] Would delete: $file"
                    else
                        echo "Deleting: $file"
                        rm "$file"
                    fi
                else
                    if [ $DRY_RUN -eq 1 ]; then
                        echo "[DRY RUN] Would keep: $file"
                    fi
                fi
            else
                # Non-integer step e.g., "last"
                if [ $DRY_RUN -eq 1 ]; then
                    echo "[DRY RUN] Would delete non-numeric checkpoint: $file"
                else
                    echo "Deleting non-numeric checkpoint: $file"
                    rm "$file"
                fi
            fi
        done
    else
        echo "Directory not found: $ckpt_dir"
    fi
done

if [ $DRY_RUN -eq 1 ]; then
    echo ""
    echo "This was a dry run. To actually delete the files, run the script with the --run argument."
fi
