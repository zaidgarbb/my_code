# semi_supervised_unet_fixmatch.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2
from glob import glob
from segmentation_models_pytorch import Unet

# ---- Dataset Classes ---- #
class LabeledMRIDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob(os.path.join(img_dir, '*.png')))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx]) / 255.0
        mask = cv2.imread(self.mask_paths[idx], 0) / 255.0
        img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img_tensor, mask_tensor

class UnlabeledMRIDataset(Dataset):
    def __init__(self, img_dir, weak_aug, strong_aug):
        self.img_paths = sorted(glob(os.path.join(img_dir, '*.png')))
        self.weak_aug = weak_aug
        self.strong_aug = strong_aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx]) / 255.0
        img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        return self.weak_aug(img_tensor), self.strong_aug(img_tensor)

# ---- Transforms ---- #
def weak_transform(x):
    return x

def strong_transform(x):
    return x + 0.1 * torch.randn_like(x)

# ---- Model ---- #
model = Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)

# ---- Loss ---- #
class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        smooth = 1.
        iflat = preds.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

# ---- Training Loop ---- #
def train_one_epoch(model, labeled_loader, unlabeled_loader, optimizer, criterion, lambda_u=0.5):
    model.train()
    total_loss = 0
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx, (labeled_imgs, labels) in enumerate(labeled_loader):
        labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
        outputs = model(labeled_imgs)
        loss_sup = criterion(outputs, labels)

        try:
            weak_imgs, strong_imgs = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            weak_imgs, strong_imgs = next(unlabeled_iter)

        weak_imgs, strong_imgs = weak_imgs.cuda(), strong_imgs.cuda()

        with torch.no_grad():
            pseudo_labels = torch.sigmoid(model(weak_imgs))
            pseudo_labels = (pseudo_labels > 0.6).float()

        outputs_u = model(strong_imgs)
        loss_unsup = criterion(outputs_u, pseudo_labels)

        loss = loss_sup + lambda_u * loss_unsup
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(labeled_loader)
