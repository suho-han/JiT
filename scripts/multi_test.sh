#!/bin/bash

set -e

# Dataset / Model / Device / Steps /  Soft_Vote
# uv run bash scripts/test.sh OCTA500_6M JiT-B/16 3 10000 True
bash scripts/test.sh MoNuSeg JiT_ParaCond-H/16 10000 1 1 dice_bce
bash scripts/test.sh MoNuSeg JiT_ParaCond-H/16 10000 1 3 dice_bce
bash scripts/test.sh MoNuSeg JiT_ParaCond-H/16 10000 1 5 dice_bce