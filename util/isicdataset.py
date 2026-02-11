import os
import random
from collections import defaultdict

import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset, Sampler

from util.transforms import ColorJitter, Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, ToTensor


def get_isic_transform(image_size):
    transform_train = Compose([
        RandomCrop((image_size, image_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        # RandomAffine(int(22), scale=(float(0.75), float(1.25))),
        ColorJitter(brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1),
        ToTensor(),
    ])

    transform_val = Compose([
        RandomCrop((image_size, image_size)),
        ToTensor(),
    ])

    transform_test = Compose([
        ToTensor(),
    ])

    return transform_train, transform_val, transform_test


class ISICSegmentationDataset(Dataset):
    """
    Custom dataset for ISIC binary segmentation.
    Assumes directory structure:
    root/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── labels/
        ├── image1.png
        ├── image2.png
        └── ...
    """

    def __init__(
        self,
        root_dir,
        image_ext='.jpg',
        label_ext='.png',
        img_size=None,
        transform=None,
    ):
        """
        Args:
            root_dir (str): Root directory containing 'images' and 'labels' subdirectories
            transform (callable, optional): Transforms to apply to images
            target_transform (callable, optional): Transforms to apply to labels
            image_ext (str): File extension for images (e.g., '.jpg')
            label_ext (str): File extension for labels (e.g., '.png')
            transform (callable, optional): Synchronized transforms for image and label
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.transform = transform
        self._image_sizes = None
        # Get list of image files
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.labels_dir):
            raise ValueError(f"Labels directory not found: {self.labels_dir}")

        self.image_files = natsorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith(image_ext)
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        if img_size is None and transform is None:
            raise ValueError("Either img_size or transform must be provided")

    def get_image_sizes(self):
        if self._image_sizes is None:
            sizes = []
            for image_fn in self.image_files:
                image_path = os.path.join(self.images_dir, image_fn)
                with Image.open(image_path) as img:
                    width, height = img.size
                sizes.append((height, width))
            self._image_sizes = sizes
        return self._image_sizes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_fn = self.image_files[idx]
        image_name = os.path.splitext(image_fn)[0]

        # Load image
        image_path = os.path.join(self.images_dir, image_fn)
        image = Image.open(image_path).convert('RGB')  # RGB for ISIC

        # Load label
        if 'ISIC2018/' in self.root_dir:
            label_path = os.path.join(self.labels_dir, image_name + "_segmentation" + self.label_ext)
        elif 'ISIC2016/' in self.root_dir:
            label_path = os.path.join(self.labels_dir, image_name + "_Segmentation" + self.label_ext)

        if os.path.exists(label_path):
            label = Image.open(label_path).convert('L')  # Grayscale
            label = np.array(label)
            # Squeeze single-channel dimension if present
            label = np.squeeze(label)
            # Convert numpy array to PIL Image
            label = Image.fromarray(label, mode='L')
        else:
            raise FileNotFoundError(f"Label not found for image {label_path}")

        # Apply joint transform (e.g., same crop/flip for image & mask)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label


class SameSizeBatchSampler(Sampler[list[int]]):
    def __init__(self, sizes, batch_size, shuffle=False, drop_last=False):
        self.sizes = list(sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        indices_by_size = defaultdict(list)
        for idx, size in enumerate(self.sizes):
            indices_by_size[size].append(idx)
        self._indices_by_size = indices_by_size

    def __iter__(self):
        batches = []
        for indices in self._indices_by_size.values():
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        count = 0
        for indices in self._indices_by_size.values():
            full, remainder = divmod(len(indices), self.batch_size)
            count += full + (0 if remainder == 0 or self.drop_last else 1)
        return count
