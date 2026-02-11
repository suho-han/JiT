#!/bin/bash

set -e

# Dataset / Model / Steps / Device / Soft_Vote
uv run bash scripts/test.sh ISIC2016 JiT-B/16 10000 0 False
uv run bash scripts/test.sh ISIC2016 JiT-B/16 10000 0 True


