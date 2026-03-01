"""
Test JiT model architecture for image-conditioned segmentation.
Tests:
- Model initialization with cond_channels and out_channels
- Forward pass with image conditioning
- Output shape verification
"""
import sys

import autorootcwd
import torch

from src.models import JiT_models

# Disable torch.compile for testing to avoid device placement issues
torch._dynamo.config.suppress_errors = True


def test_jit_basic_init(model_name='JiT-B/16'):
    """Test basic JiT initialization."""
    print("Testing JiT basic initialization...")
    model = JiT_models[model_name](
        input_size=256,
        in_channels=1,
        cond_channels=1,
        out_channels=1,
        attn_drop=0.0,
        proj_drop=0.0,
    )
    print(f"✓ Model created: {model}")
    print(f"  - in_channels: {model.in_channels}")
    print(f"  - cond_channels: {model.cond_channels}")
    print(f"  - out_channels: {model.out_channels}")
    return model


def test_jit_forward_pass(model):
    """Test forward pass with image conditioning."""
    print("\nTesting JiT forward pass...")
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Create dummy inputs
    mask_input = torch.randn(batch_size, model.in_channels, 256, 256, device=device)
    timesteps = torch.tensor([0.5] * batch_size, device=device)
    cond_image = torch.randn(batch_size, model.cond_channels, 256, 256, device=device)

    with torch.no_grad():
        output = model(mask_input, timesteps, cond_image)

    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {mask_input.shape}")
    print(f"  - Condition shape: {cond_image.shape}")
    print(f"  - Output shape: {output.shape}")
    assert output.shape == (batch_size, model.out_channels, 256, 256), f"Unexpected output shape: {output.shape}"
    print(f"✓ Output shape correct: {output.shape}")


def test_jit_without_conditioning(model):
    """Test forward pass without image conditioning (cond=None)."""
    print("\nTesting JiT forward pass without conditioning...")
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Create dummy inputs without conditioning
    mask_input = torch.randn(batch_size, model.in_channels, 256, 256, device=device)
    timesteps = torch.tensor([0.5] * batch_size, device=device)

    with torch.no_grad():
        output = model(mask_input, timesteps, cond=None)

    print(f"✓ Forward pass without conditioning successful")
    print(f"  - Output shape: {output.shape}")
    assert output.shape == (batch_size, model.out_channels, 256, 256)


def test_jit_b16_factory():
    """Test JiT-B/16 factory function."""
    print("\nTesting JiT-B/16 factory function...")
    model = JiT_models['JiT-B/16'](
        input_size=256,
        in_channels=1,
        cond_channels=1,
        out_channels=1,
    )
    print(f"✓ JiT-B/16 created")
    print(f"  - Hidden size: {model.hidden_size}")
    print(f"  - Depth: 12")
    print(f"  - Num heads: 12")

    # Test forward pass
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    mask_input = torch.randn(batch_size, 1, 256, 256, device=device)
    timesteps = torch.tensor([0.5], device=device)
    cond_image = torch.randn(batch_size, 1, 256, 256, device=device)

    with torch.no_grad():
        output = model(mask_input, timesteps, cond_image)

    print(f"✓ JiT-B/16 forward pass successful")
    print(f"  - Output shape: {output.shape}")

def test_cond_weights():
    """Test varying cond_weight configurations for JiT_ParaCondWave and JiT_ParaCond."""
    print("\nTesting cond_weights configurations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test for JiT_ParaCondWave
    for weight_mode in ['fixed', 'learnable', 'shared', 'zero_init']:
        print(f"  - Testing JiT_ParaCondWave with {weight_mode}")
        cond_weight = {'cond': weight_mode, 'low_cond': weight_mode, 'high_cond': weight_mode}
        model = JiT_models['JiT_ParaCondWave-B/16'](
            input_size=64,
            in_channels=1,
            cond_channels=1,
            out_channels=1,
            cond_weight=cond_weight
        ).to(device)
        
        mask_input = torch.randn(1, 1, 64, 64, device=device)
        timesteps = torch.tensor([0.5], device=device)
        cond_image = torch.randn(1, 1, 64, 64, device=device)
        
        # Test forward pass works without errors
        output = model(mask_input, timesteps, cond_image)
        assert output.shape == (1, 1, 64, 64)
        
        if weight_mode == 'shared':
            assert hasattr(model, 'shared_cond_w')
        if weight_mode == 'learnable' or weight_mode == 'zero_init':
            assert hasattr(model.blocks[0], 'cond_w')

    # Test for JiT_ParaCond 
    for weight_mode in ['fixed', 'learnable', 'shared', 'zero_init']:
        print(f"  - Testing JiT_ParaCond with {weight_mode}")
        cond_weight = {'cond': weight_mode}
        model = JiT_models['JiT_ParaCond-B/16'](
            input_size=64,
            in_channels=1,
            cond_channels=1,
            out_channels=1,
            cond_weight=cond_weight
        ).to(device)
        
        mask_input = torch.randn(1, 1, 64, 64, device=device)
        timesteps = torch.tensor([0.5], device=device)
        cond_image = torch.randn(1, 1, 64, 64, device=device)
        
        # Test forward pass works without errors
        output = model(mask_input, timesteps, cond_image)
        assert output.shape == (1, 1, 64, 64)
        
        if weight_mode == 'shared':
            assert hasattr(model, 'shared_cond_w')
        if weight_mode == 'learnable' or weight_mode == 'zero_init':
            assert hasattr(model.blocks[0], 'cond_w')
    print("✓ cond_weight tests passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing JiT Model Architecture")
    print("=" * 60)
    model_names = ['JiT-B/16', 'JiT_CondImg-B/16', 'JiT_ParaCond-B/16', 'JiT_ParaCondWave-B/16']
    for name in model_names:
        print(f"\nTesting model: {name}")
        model = test_jit_basic_init(name)
        test_jit_forward_pass(model)
        test_jit_without_conditioning(model)

    test_jit_b16_factory()
    test_cond_weights()

    print("=" * 60)
    print("JiT model architecture verified! ✓")
    print("=" * 60)
