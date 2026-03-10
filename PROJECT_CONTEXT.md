# JiT Segmentation Project Context & Plan

## Current Status

- **Task:** Transitioning JiT (Just image Transformer) from ImageNet generation to Medical Image Segmentation.
- **Datasets:** OCTA500_6M, MoNuSeg, ISIC2016, ISIC2018.
- **Models:** JiT-B, JiT-L, JiT-H with Image-Conditioning, Parameter-Conditioning, Wavelet-Conditioning, ParaCondFilm, and ParaCondWaveFix variants.
- **Metrics:** Dice, IoU, Sensitivity, Specificity, HD95, AJI (for ISIC), clDice (for OCTA).

## Improvements Implemented

### 1. Multi-run Ensemble Inference (Completed)

- **Goal:** Improve Dice score and IoU by averaging multiple diffusion samples.
- **Implementation:** Added `--num_samples` to `src/inference_jit.py`. Sampling results are averaged before binarization.
- **Usage:** `uv run python src/inference_jit.py ... --num_samples 5`

### 2. Hybrid & Weighted Loss Functions (Completed)

- **Goal:** Optimize training for binary segmentation and handle class imbalance.
- **Implementation:**
  - `SoftDiceLoss`: Differentiable dice loss for training.
  - `WeightedBCEDiceLoss`: Allows balancing BCE and Dice components (e.g., 1:2 ratio).
- **Usage:** Set `--add_loss --add_loss_name weighted_dice_bce` in training scripts.

### 3. New Conditioning Variants (Completed)

- **Goal:** Add stronger yet stable image conditioning without modifying existing model implementations.
- **Implementation:**
  - `JiT_ParaCondFiLM` (`src/models/JiT_ParaCondFiLM.py`): ParaCond-style conditioning with an additional FiLM modulation stage on cross-attended condition features.
  - `JiT_ParaCondWaveFix` (`src/models/JiT_paracondwavefix.py`): ParaCondWave-style conditioning with softmax-normalized stream weights (`cond`, `low_cond`, `high_cond`) to reduce scale collapse.
  - Model registry updated in `src/models/__init__.py`.
- **Usage examples:**
  - `JiT_ParaCondFiLM-B/16`
  - `JiT_ParaCondWaveFix-B/16`

## Future Roadmap

### 4. Model Architecture Refinement (Planned)

- **Skip Connections:** Implement U-Net like skip connections between encoder and decoder blocks in `JiT_paracondwave.py` to preserve fine-grained details.
- **Cross-Attention Conditioning:** Replace simple concatenation/addition of image conditions with Cross-Attention modules for better feature fusion.
- **Hierarchical Features:** Explore multi-scale patch embeddings to capture both global context and local details.

### 5. Advanced Training Strategies

- **Boundary Loss:** Integrate boundary-aware loss functions to improve edge precision.
- **clDice Integration:** Better utilize `clDice` in the training loop for vascular datasets (OCTA).

## Experiment Tracking

- Checkpoints are saved in `outputs/` with naming convention: `{Model}-{Dataset}-{Loss}/checkpoints/`.
- Logs are available in `outputs/logs/`.
- Performance metrics are saved as CSV files in the respective output directories after inference.
