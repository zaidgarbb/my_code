import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import nibabel as nib
import random
from torchvision.transforms import functional as F

def load_brats_image(file_path):
    """Load a single MRI modality from a NIfTI file."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    # Important: Handle the shape; this assumes you want a 2D slice.
    if len(img_data.shape) > 3:
        img_data = img_data[:, :, 0]  # Or any other slicing logic
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())  # Normalize to [0, 1]
    return torch.from_numpy(img_data).float()

class LabeledDataset(Dataset):
    def __init__(self, image_dir, mask_dir, modalities=('t1', 't1ce', 't2', 'flair'), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.modalities = modalities
        self.transform = transform
        self.image_paths = {}
        for mod in modalities:
            self.image_paths[mod] = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(f'{mod}.nii.gz')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        assert len(self.mask_paths) == len(self.image_paths[modalities[0]]), "Number of images and masks should match"

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        images = []
        for mod in self.modalities:
            img = load_brats_image(self.image_paths[mod][idx])
            images.append(img)
        img = torch.stack(images, dim=0)  # Shape: (num_modalities, H, W)

        mask = nib.load(self.mask_paths[idx])
        mask_data = mask.get_fdata()
        if len(mask_data.shape) > 3:
            mask_data = mask_data[:,:,0]
        mask_np = np.array(mask_data, dtype=np.uint8)
        label = 1 if mask_np.max() > 0 else 0

        if self.transform:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, modalities=('t1', 't1ce', 't2', 'flair'), weak_transform=None, strong_transform=None):
        self.image_dir = image_dir
        self.modalities = modalities
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.image_paths = {}
        for mod in modalities:
            self.image_paths[mod] = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(f'{mod}.nii.gz')])

    def __len__(self):
        return len(self.image_paths[self.modalities[0]])

    def __getitem__(self, idx):
        images = []
        for mod in self.modalities:
            img = load_brats_image(self.image_paths[mod][idx])
            images.append(img)
        img = torch.stack(images, dim=0)

        weak_image = self.weak_transform(img) if self.weak_transform else img
        strong_image = self.strong_transform(img) if self.strong_transform else img
        return weak_image, strong_image

def get_augmentations():
    weak = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    strong = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])
    return weak, strong
