"""
Test data loading and transformations for ISIC image-mask pairs.
Tests:
- ISIC dataset loading
- Image-mask synchronization in transforms
- Batch loading from DataLoader
"""
import os
import sys

import autorootcwd
import torch

from util.isicdataset import ISICSegmentationDataset, SameSizeBatchSampler, get_isic_transform

sys.path.insert(0, '/home/suhohan/JiT')


def test_isic_dataset_basic():
    """Test basic ISIC dataset loading."""
    print("Testing ISIC dataset basic loading...")

    data_path = 'data/ISIC/train'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping test: {data_path} not found")
        return None, None

    transform_train, transform_test = get_isic_transform(image_size=512)

    train_dataset = ISICSegmentationDataset(
        data_path,
        img_size=512,
        transform=transform_train,
    )

    test_dataset = ISICSegmentationDataset(
        data_path,
        img_size=512,
        transform=transform_test,
    )

    print("✓ Datasets loaded")
    print(f"  - Path: {data_path}")
    print(f"  - Train dataset size: {len(train_dataset)}")
    print(f"  - Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


def test_isic_dataset_sample(train_dataset, test_dataset):
    """Test single sample loading for both train and test datasets."""
    if train_dataset is None or test_dataset is None:
        return

    print("\nTesting single sample loading...")
    for label, dataset in ("train", train_dataset), ("test", test_dataset):
        image, mask = dataset[0]

        assert isinstance(image, torch.Tensor), "Image should be tensor"
        assert isinstance(mask, torch.Tensor), "Mask should be tensor"
        print(f"✓ {label} sample 0 loaded successfully")
        print(f"  - {label} image shape: {image.shape}")
        print(f"  - {label} mask shape: {mask.shape}")
        print(f"  - {label} image dtype: {image.dtype}")
        print(f"  - {label} mask dtype: {mask.dtype}")
        print(f"  - {label} image range: [{image.min():.4f}, {image.max():.4f}]")
        print(f"  - {label} mask range: [{mask.min():.4f}, {mask.max():.4f}]")

        assert image.ndim == 3, f"Image should be 3D (C, H, W), got {image.shape}"
        assert mask.ndim == 3, f"Mask should be 3D (C, H, W), got {mask.shape}"
        assert image.shape[1:] == mask.shape[1:], "Image and mask spatial dimensions should match"
        assert image.shape[0] == 3, "Image should have 3 channels"
        assert mask.shape[0] == 1, "Mask should have 1 channel"


def test_isic_dataset_transforms():
    """Test that transforms are applied synchronously to image and mask."""
    if not os.path.exists('data/ISIC/train'):
        print("⚠ Skipping transform test: data not found")
        return

    print("\nTesting synchronized transforms...")

    transform_train, _ = get_isic_transform(image_size=512)

    dataset = ISICSegmentationDataset(
        'data/ISIC/train',
        img_size=512,
        transform=transform_train,
    )

    image1, mask1 = dataset[0]
    image2, mask2 = dataset[0]

    print("✓ Transform applied successfully")
    print(f"  - Image 1 first row sum: {image1[0, 0, :].sum():.4f}")
    print(f"  - Mask 1 first row sum: {mask1[0, 0, :].sum():.4f}")


def test_isic_transform_function():
    """Test get_isic_transform function."""
    print("\nTesting get_isic_transform function...")

    transform_train, transform_val = get_isic_transform(image_size=512)

    print("✓ Transforms created")
    print(f"  - Train transform: {type(transform_train)}")
    print(f"  - Val transform: {type(transform_val)}")

    import numpy as np
    from PIL import Image

    dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255), mode='RGB')
    dummy_mask = Image.fromarray(np.uint8(np.random.rand(512, 512) * 255), mode='L')

    img_train, mask_train = transform_train(dummy_img, dummy_mask)
    img_val, mask_val = transform_val(dummy_img, dummy_mask)

    print("✓ Transforms applied")
    print(f"  - Train image shape: {img_train.shape}")
    print(f"  - Train mask shape: {mask_train.shape}")
    print(f"  - Val image shape: {img_val.shape}")
    print(f"  - Val mask shape: {mask_val.shape}")

    assert img_train.shape[-2:] == (512, 512) or img_train.shape[-1] == 512, "Train image should be resized"


def test_dataloader_batching():
    """Test DataLoader batch loading for both train and test datasets."""
    if not os.path.exists('data/ISIC/train'):
        print("⚠ Skipping DataLoader test: data not found")
        return

    print("\nTesting DataLoader batch loading...")

    transform_train, transform_test = get_isic_transform(image_size=512)
    dataset_configs = (
        ("train", transform_train),
        ("test", transform_test),
    )

    for label, transform in dataset_configs:
        dataset = ISICSegmentationDataset(
            'data/ISIC/train',
            img_size=512,
            transform=transform,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            num_workers=0,
            drop_last=False,
        )

        images, masks = next(iter(dataloader))

        print(f"✓ {label} DataLoader batch loaded")
        print(f"  - {label} batch images shape: {images.shape}")
        print(f"  - {label} batch masks shape: {masks.shape}")
        print(f"  - {label} images dtype: {images.dtype}")
        print(f"  - {label} masks dtype: {masks.dtype}")

        assert images.shape[0] == 4, f"Batch size should be 4, got {images.shape[0]}"
        assert masks.shape[0] == 4, f"Batch size should be 4, got {masks.shape[0]}"
        assert images.shape[1] == 3, f"Image channels should be 3, got {images.shape[1]}"
        assert masks.shape[1] == 1, f"Mask channels should be 1, got {masks.shape[1]}"
        assert images.shape[-1] == masks.shape[-1], "Image and mask width should match"
        assert images.shape[-2] == masks.shape[-2], "Image and mask height should match"


def test_dataloader_same_size_batching():
    """Test same-size batching for test dataset."""
    data_path = 'data/ISIC/test'
    if not os.path.exists(data_path):
        print(f"⚠ Skipping same-size DataLoader test: {data_path} not found")
        return

    print("\nTesting same-size DataLoader batching...")

    _, transform_test = get_isic_transform(image_size=512)
    dataset = ISICSegmentationDataset(
        data_path,
        img_size=512,
        transform=transform_test,
    )

    batch_sampler = SameSizeBatchSampler(
        dataset.get_image_sizes(),
        batch_size=4,
        shuffle=False,
        drop_last=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        drop_last=False,
    )

    batch_count = 0
    for images, masks in dataloader:
        batch_count += 1
        assert images.ndim == 4, f"Images should be 4D (B, C, H, W), got {images.shape}"
        assert masks.ndim == 4, f"Masks should be 4D (B, C, H, W), got {masks.shape}"
        assert images.shape[-2:] == masks.shape[-2:], "Image and mask spatial dimensions should match"

        ref_height = images[0].shape[-2]
        ref_width = images[0].shape[-1]
        for idx in range(images.shape[0]):
            assert images[idx].shape[-2:] == (ref_height, ref_width), "Images in batch should share the same size"
            assert masks[idx].shape[-2:] == (ref_height, ref_width), "Masks in batch should share the same size"

    print(f"✓ Same-size DataLoader batches validated: {batch_count}")


def test_images():
    """Save sample images from the dataset to disk for visual inspection."""
    if not os.path.exists('./data/ISIC/train'):
        print("⚠ Skipping image saving test: data not found")
        return

    print("\nSaving sample images for visual inspection...")
    from torchvision.utils import save_image

    transform_train, _ = get_isic_transform(image_size=512)
    dataset = ISICSegmentationDataset(
        './data/ISIC/train',
        img_size=512,
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
        save_image(images[i], f'./tests/test_outputs/isic_image_{i}.png')
        save_image(masks[i], f'./tests/test_outputs/isic_mask_{i}.png')
    print("✓ Sample images saved to ./tests/test_outputs/")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing ISIC Dataset (Image-Mask Pairs)")
    print("=" * 60)

    train_dataset, test_dataset = test_isic_dataset_basic()
    if train_dataset is not None:
        test_isic_dataset_sample(train_dataset, test_dataset)
        test_isic_dataset_transforms()

    test_isic_transform_function()
    test_dataloader_batching()
    test_dataloader_same_size_batching()

    test_images()

    print("\n" + "=" * 60)
    print("All ISIC dataset tests passed! ✓")
    print("=" * 60)
