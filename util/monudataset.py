import os

import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset

from util.transforms import ColorJitter, Compose, Normalize, RandomAffine, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToPILImage, ToTensor


def get_monu_transform(image_size):

    transform_train = Compose([
        Resize((512, 512)),
        RandomCrop((image_size, image_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomAffine(int(22), scale=(float(0.75), float(1.25))),
        ColorJitter(brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1),
        ToTensor(),
        Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    transform_test = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    return transform_train, transform_test


class MoNuSegmentationDataset(Dataset):
    """
    Custom dataset for MoNuSeg binary segmentation.
    Assumes directory structure:
    root/
    ├── images/
    │   ├── image1.bmp
    │   ├── image2.bmp
    │   └── ...
    └── labels/
        ├── image1.bmp
        ├── image2.bmp
        └── ...
    """

    def __init__(
        self,
        root_dir,
        image_ext='.tif',
        label_ext='.npy',
        img_size=None,
        transform=None,
    ):
        """
        Args:
            root_dir (str): Root directory containing 'images' and 'labels' subdirectories
            transform (callable, optional): Transforms to apply to images
            target_transform (callable, optional): Transforms to apply to labels
            image_ext (str): File extension for images (e.g., '.bmp')
            label_ext (str): File extension for labels (e.g., '.bmp')
            transform (callable, optional): Synchronized transforms for image and label
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.transform = transform
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_fn = self.image_files[idx]
        image_name = os.path.splitext(image_fn)[0]

        # Load image
        image_path = os.path.join(self.images_dir, image_fn)
        image = Image.open(image_path).convert('RGB')  # RGB for MoNuSeg

        # Load label
        label_path = os.path.join(self.labels_dir, image_name + self.label_ext)

        if os.path.exists(label_path):
            label = np.load(label_path)
            # Squeeze single-channel dimension if present
            label = np.squeeze(label)
            # Convert numpy array to PIL Image
            label = Image.fromarray(label, mode='L')
        else:
            raise FileNotFoundError(f"Label not found for image {image_fn}")

        # Apply joint transform (e.g., same crop/flip for image & mask)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label
