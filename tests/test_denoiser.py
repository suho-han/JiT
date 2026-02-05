"""
Test Denoiser for image-conditioned diffusion model.
Tests:
- Denoiser initialization with image conditioning
- Forward pass (training loss computation)
- Generate method (sampling)
"""
import sys

import torch

sys.path.insert(0, '/home/suhohan/JiT')

from argparse import Namespace

from denoiser import Denoiser


def create_test_args():
    """Create test arguments."""
    return Namespace(
        model='JiT-B/16',
        img_size=256,
        img_channel=1,  # Input image conditioning
        mask_channel=1,  # Output mask target
        attn_dropout=0.0,
        proj_dropout=0.0,
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=5e-2,
        # EMA parameters
        ema_decay1=0.9999,
        ema_decay2=0.9996,
        # Sampling parameters (kept for compatibility)
        sampling_method='heun',
        num_sampling_steps=10,
        cfg=1.0,
        interval_min=0.0,
        interval_max=1.0,
    )


def test_denoiser_init():
    """Test Denoiser initialization."""
    print("Testing Denoiser initialization...")
    args = create_test_args()
    denoiser = Denoiser(args)
    
    print(f"✓ Denoiser created")
    print(f"  - Model: {args.model}")
    print(f"  - Image size: {args.img_size}")
    print(f"  - Image channel (condition): {args.img_channel}")
    print(f"  - Mask channel (target): {args.mask_channel}")
    print(f"  - Sampling method: {args.sampling_method}")
    print(f"  - Num sampling steps: {args.num_sampling_steps}")
    return denoiser


def test_denoiser_forward(denoiser):
    """Test Denoiser forward pass (training)."""
    print("\nTesting Denoiser forward pass (training)...")
    denoiser.train()
    device = 'cpu'
    denoiser.to(device)
    
    batch_size = 2
    mask_target = torch.randn(batch_size, 1, 256, 256, device=device)
    image_cond = torch.randn(batch_size, 1, 256, 256, device=device)
    
    # Forward pass computes loss
    loss = denoiser(mask_target, image_cond)
    
    print(f"✓ Forward pass successful")
    print(f"  - Mask target shape: {mask_target.shape}")
    print(f"  - Image condition shape: {image_cond.shape}")
    print(f"  - Loss: {loss.item():.6f}")
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_denoiser_generate(denoiser):
    """Test Denoiser generate method (sampling)."""
    print("\nTesting Denoiser generate method (sampling)...")
    denoiser.eval()
    device = 'cpu'
    denoiser.to(device)
    
    batch_size = 1
    image_cond = torch.randn(batch_size, 1, 256, 256, device=device)
    
    with torch.no_grad():
        sampled_masks = denoiser.generate(image_cond)
    
    print(f"✓ Generate successful")
    print(f"  - Condition shape: {image_cond.shape}")
    print(f"  - Sampled mask shape: {sampled_masks.shape}")
    assert sampled_masks.shape == (batch_size, 1, 256, 256), f"Unexpected output shape: {sampled_masks.shape}"
    print(f"  - Sampled mask range: [{sampled_masks.min():.4f}, {sampled_masks.max():.4f}]")


def test_denoiser_ema_update(denoiser):
    """Test EMA parameter update."""
    print("\nTesting EMA parameter update...")
    denoiser.train()
    device = 'cpu'
    denoiser.to(device)
    
    # Initialize EMA params
    denoiser.ema_params1 = [p.clone().to(device) for p in denoiser.parameters()]
    denoiser.ema_params2 = [p.clone().to(device) for p in denoiser.parameters()]
    
    # Update EMA
    denoiser.update_ema()
    
    print(f"✓ EMA update successful")
    print(f"  - EMA params 1 count: {len(denoiser.ema_params1)}")
    print(f"  - EMA params 2 count: {len(denoiser.ema_params2)}")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Denoiser (Image-Conditioned)")
    print("=" * 60)
    
    denoiser = test_denoiser_init()
    test_denoiser_forward(denoiser)
    test_denoiser_generate(denoiser)
    test_denoiser_ema_update(denoiser)
    
    print("\n" + "=" * 60)
    print("All Denoiser tests passed! ✓")
    print("=" * 60)
