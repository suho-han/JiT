# JiT (Just image Transformer) for Pixel-space Diffusion

## Project Overview
This project is a PyTorch/GPU re-implementation and adaptation of the "Just image Transformer" (JiT) model for pixel-space diffusion, originally proposed in the paper [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720). While the original work focused on high-resolution image generation (e.g., ImageNet), this repository has been specifically adapted for **medical image segmentation** (e.g., OCTA, MoNuSeg, ISIC).

**For detailed project goals, roadmap, and recent improvements (Multi-run Ensemble, Hybrid Loss), please refer to [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md).**

### Key Technologies
- **Framework:** PyTorch
- **Package Manager:** `uv` (preferred) or `conda`
- **Model Architecture:** Transformer-based diffusion (JiT)
- **Task:** Image-to-Mask diffusion (predicting segmentation masks conditioned on images)
- **Key Libraries:** `timm`, `einops`, `torch-fidelity`, `scipy`, `opencv-python`

### Architecture
- **`src/models/`**: Contains various JiT model variants (Base, Image-conditioned, Parameter-conditioned, Wavelet-conditioned).
- **`src/denoiser.py`**: Implements the diffusion denoising process.
- **`src/engine_jit.py`**: Training and evaluation logic, including segmentation metrics (Dice, IoU, HD95).
- **`src/main_jit.py`**: Entry point for training.
- **`src/inference_jit.py`**: Entry point for inference and evaluation.

## Building and Running

### Environment Setup
The project uses `uv` for dependency management.
```bash
# Install dependencies using uv
uv sync

# Or using conda
conda env create -f environment.yaml
conda activate jit
```

### Training
A unified script `scripts/run_jit.sh` is provided for training on different datasets.
```bash
# Usage: uv run bash scripts/run_jit.sh <dataset> <model> <device>
uv run bash scripts/run_jit.sh OCTA500_6M JiT_ParaCondWave-B/16 0
```

### Testing and Evaluation
Inference can be run via `src/inference_jit.py`. **Multi-run ensemble** can be enabled via `--num_samples`.
```bash
uv run python src/inference_jit.py --model <model_name> --checkpoint <path_to_ckpt> \
--dataset <dataset_name> --data_path <test_data_path> --output_dir outputs \
--num_samples 5 --metrics --device cuda
```

### Running Tests
A comprehensive test suite is available in the `tests/` directory.
```bash
# Run all tests
bash tests/run_tests.sh
```

## Development Conventions
- **Source Code:** Core logic is in the `src/` directory; utilities in `util/`.
- **Data/Outputs:** Datasets in `data/`, checkpoints/logs in `outputs/`.
- **Testing:** New features MUST be verified using scripts in the `tests/` directory.
- **Plan:** Follow the roadmap and milestones outlined in `PROJECT_CONTEXT.md`.
