#!/bin/bash

set -e

# Dataset / Model / Device / Steps /  Soft_Vote
# uv run bash scripts/test.sh OCTA500_6M JiT-B/16 3 10000 True
uv run bash scripts/test.sh MoNuSeg JiT_CondImg-B/16 3 10000
uv run bash scripts/test.sh MoNuSeg JiT_CondImg-B/16 3 20000

