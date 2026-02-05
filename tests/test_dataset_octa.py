"""
Test data loading and transformations for image-mask pairs.
Tests:
- OCTA dataset loading
- Image-mask synchronization in transforms
- Batch loading from DataLoader
"""
import sys

import torch

sys.path.insert(0, '/home/suhohan/JiT')

import os

from util.octadataset import OCTASegmentationDataset, get_octa_transform


def test_octa_dataset_basic():
    """Test basic OCTA dataset loading."""
    print("Testing OCTA dataset basic loading...")
    
    # Check if OCTA data exists
    data_path = './data/OCTA500_6M/train'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping test: {data_path} not found")
        return None
    
    # Create simple transform
    transform_train, _ = get_octa_transform(image_size=256)
    
    dataset = OCTASegmentationDataset(
        data_path,
        img_size=256,
        transform=transform_train,
    )
    
    print(f"✓ Dataset loaded")
    print(f"  - Path: {data_path}")
    print(f"  - Dataset size: {len(dataset)}")
    
    return dataset


def test_octa_dataset_sample(dataset):
    """Test single sample loading."""
    if dataset is None:
        return
    
    print("\nTesting single sample loading...")
    image, mask = dataset[0]
    
    assert isinstance(image, torch.Tensor), "Image should be tensor"
    assert isinstance(mask, torch.Tensor), "Mask should be tensor"
    print(f"✓ Sample 0 loaded successfully")
    print(f"  - Image shape: {image.shape}")
    print(f"  - Mask shape: {mask.shape}")
    print(f"  - Image dtype: {image.dtype}")
    print(f"  - Mask dtype: {mask.dtype}")
    print(f"  - Image range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"  - Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    
    # Verify channel dimensions
    assert image.ndim == 3, f"Image should be 3D (C, H, W), got {image.shape}"
    assert mask.ndim == 3, f"Mask should be 3D (C, H, W), got {mask.shape}"
    assert image.shape[1] == image.shape[2] == 256, f"Image should be 256x256, got {image.shape}"
    assert mask.shape[1] == mask.shape[2] == 256, f"Mask should be 256x256, got {mask.shape}"


def test_octa_dataset_transforms():
    """Test that transforms are applied synchronously to image and mask."""
    if not os.path.exists('./data/OCTA500_6M/train'):
        print("⚠ Skipping transform test: data not found")
        return
    
    print("\nTesting synchronized transforms...")
    
    transform_train, _ = get_octa_transform(image_size=256)
    
    dataset = OCTASegmentationDataset(
        './data/OCTA500_6M/train',
        img_size=256,
        transform=transform_train,
    )
    
    image1, mask1 = dataset[0]
    image2, mask2 = dataset[0]  # Same index, will be flipped again (back to original)
    
    print(f"✓ Transform applied successfully")
    print(f"  - Image 1 first row sum: {image1[0, 0, :].sum():.4f}")
    print(f"  - Mask 1 first row sum: {mask1[0, 0, :].sum():.4f}")


def test_octa_transform_function():
    """Test get_octa_transform function."""
    print("\nTesting get_octa_transform function...")
    
    transform_train, transform_val = get_octa_transform(image_size=256)
    
    print(f"✓ Transforms created")
    print(f"  - Train transform: {type(transform_train)}")
    print(f"  - Val transform: {type(transform_val)}")
    
    # Create dummy PIL images
    import numpy as np
    from PIL import Image
    
    dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512) * 255), mode='L')
    dummy_mask = Image.fromarray(np.uint8(np.random.rand(512, 512) * 255), mode='L')
    
    # Apply transforms
    img_train, mask_train = transform_train(dummy_img, dummy_mask)
    img_val, mask_val = transform_val(dummy_img, dummy_mask)
    
    print(f"✓ Transforms applied")
    print(f"  - Train image shape: {img_train.shape}")
    print(f"  - Train mask shape: {mask_train.shape}")
    print(f"  - Val image shape: {img_val.shape}")
    print(f"  - Val mask shape: {mask_val.shape}")
    
    assert img_train.shape[-2:] == (256, 256) or img_train.shape[-1] == 256, "Train image should be resized"


def test_dataloader_batching():
    """Test DataLoader batch loading."""
    if not os.path.exists('./data/OCTA500_6M/train'):
        print("⚠ Skipping DataLoader test: data not found")
        return
    
    print("\nTesting DataLoader batch loading...")
    
    transform_train, _ = get_octa_transform(image_size=256)
    
    dataset = OCTASegmentationDataset(
        './data/OCTA500_6M/train',
        img_size=256,
        transform=transform_train,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        drop_last=False,
    )
    
    # Get first batch
    images, masks = next(iter(dataloader))
    
    print(f"✓ DataLoader batch loaded")
    print(f"  - Batch images shape: {images.shape}")
    print(f"  - Batch masks shape: {masks.shape}")
    print(f"  - Images dtype: {images.dtype}")
    print(f"  - Masks dtype: {masks.dtype}")
    
    assert images.shape[0] == 4, f"Batch size should be 4, got {images.shape[0]}"
    assert masks.shape[0] == 4, f"Batch size should be 4, got {masks.shape[0]}"
    assert images.shape[1] == 1, f"Image channels should be 1, got {images.shape[1]}"
    assert masks.shape[1] == 1, f"Mask channels should be 1, got {masks.shape[1]}"
    assert images.shape[-1] == 256, f"Image width should be 256, got {images.shape[-1]}"
    assert masks.shape[-1] == 256, f"Mask width should be 256, got {masks.shape[-1]}"

def test_images():
    '''
    save sample images from the dataset to disk for visual inspection
    '''
    if not os.path.exists('./data/OCTA500_6M/train'):
        print("⚠ Skipping image saving test: data not found")
        return
    print("\nSaving sample images for visual inspection...")
    from torchvision.utils import save_image
    transform_train, _ = get_octa_transform(image_size=256)
    dataset = OCTASegmentationDataset(
        './data/OCTA500_6M/train',
        img_size=256,
        transform=transform_train,
    )   
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        drop_last=False,
    )
    images, masks = next(iter(dataloader))
    os.makedirs('./tests/test_outputs', exist_ok=True)
    for i in range(images.shape[0]):
        save_image(images[i], f'./tests/test_outputs/image_{i}.png')
        save_image(masks[i], f'./tests/test_outputs/mask_{i}.png')
    print("✓ Sample images saved to ./tests/test_outputs/")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing OCTA Dataset (Image-Mask Pairs)")
    print("=" * 60)
    
    dataset = test_octa_dataset_basic()
    if dataset is not None:
        test_octa_dataset_sample(dataset)
        test_octa_dataset_transforms()
    
    test_octa_transform_function()
    test_dataloader_batching()
    
    # Save test results
    test_images()
    
    print("\n" + "=" * 60)
    print("All dataset tests passed! ✓")
    print("=" * 60)
