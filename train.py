import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from dataloader import LabeledDataset, UnlabeledDataset, get_augmentations
import os
import logging
import numpy as np
import random

# ✅ CONFIGURATION
BASE_PATH = '/home/ubuntu/semi_supervised_fixmatch_all_scripts/BraTS_FixMatch/BraTS_FixMatch'
LABELED_IMAGES_DIR = os.path.join(BASE_PATH, 'labeled', 'images')
LABELED_MASKS_DIR = os.path.join(BASE_PATH, 'labeled', 'masks')
UNLABELED_IMAGES_DIR = os.path.join(BASE_PATH, 'unlabeled', 'images')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'fixmatch_model.pth')
LOG_FILE = os.path.join(BASE_PATH, 'training.log')


BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
PSEUDO_LABEL_THRESHOLD = 0.95
UNSUPERVISED_LOSS_WEIGHT = 1.0
VALIDATION_SPLIT = 0.2

# ✅ REPRODUCIBILITY
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ✅ LOGGING SETUP
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print(f"🚀 Training started... Logs will be written to {LOG_FILE}")
logging.info("Starting FixMatch training...")

# ✅ DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
logging.info(f"Using device: {device}")

# ✅ AUGMENTATIONS
weak_transform, strong_transform = get_augmentations()

# ✅ DATASETS AND DATALOADERS
print("✅ Preparing datasets and dataloaders...")
logging.info("Preparing datasets and dataloaders...")

labeled_dataset = LabeledDataset(LABELED_IMAGES_DIR, LABELED_MASKS_DIR, transform=weak_transform)
unlabeled_dataset = UnlabeledDataset(UNLABELED_IMAGES_DIR, weak_transform=weak_transform, strong_transform=strong_transform)

# ✅ SPLITTING LABELED DATASET
train_size = int((1 - VALIDATION_SPLIT) * len(labeled_dataset))
val_size = len(labeled_dataset) - train_size
labeled_train_dataset, labeled_val_dataset = random_split(labeled_dataset, [train_size, val_size])

labeled_train_loader = DataLoader(labeled_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
labeled_val_loader = DataLoader(labeled_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Labeled dataset samples: {len(labeled_dataset)}")
print(f"Unlabeled dataset samples: {len(unlabeled_dataset)}")
print(f"Labeled train loader batches: {len(labeled_train_loader)}")
print(f"Labeled val loader batches: {len(labeled_val_loader)}")
print(f"Unlabeled loader batches: {len(unlabeled_loader)}")

# ✅ MODEL SETUP
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ HELPER FUNCTION
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == targets).sum().item() / targets.size(0)

# ✅ TRAINING LOOP
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    print(f"🚧 Epoch {epoch + 1}/{EPOCHS} in progress...")
    model.train()
    labeled_iter = iter(labeled_train_loader)
    unlabeled_iter = iter(unlabeled_loader)

    total_loss = 0
    for batch_idx in range(min(len(labeled_train_loader), len(unlabeled_loader))):
        print(f"🟢 Processing batch {batch_idx + 1}/{min(len(labeled_train_loader), len(unlabeled_loader))}")

        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_train_loader)
            x_l, y_l = next(labeled_iter)

        try:
            x_weak, x_strong = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_weak, x_strong = next(unlabeled_iter)

        x_l, y_l = x_l.to(device), y_l.view(-1).long().to(device)
        x_weak, x_strong = x_weak.to(device), x_strong.to(device)

        # Supervised loss
        outputs_l = model(x_l)
        loss_sup = criterion(outputs_l, y_l)

        # Unsupervised loss
        with torch.no_grad():
            pseudo_labels = torch.softmax(model(x_weak), dim=1)
            max_probs, targets_u = torch.max(pseudo_labels, dim=1)
            mask = (max_probs >= PSEUDO_LABEL_THRESHOLD).float()

        outputs_u = model(x_strong)
        loss_unsup = (criterion(outputs_u, targets_u) * mask).mean()

        # Total loss
        loss = loss_sup + UNSUPERVISED_LOSS_WEIGHT * loss_unsup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(labeled_train_loader)
    print(f"✅ Epoch {epoch + 1}/{EPOCHS} completed with Avg Loss: {avg_loss:.4f}")
    logging.info(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✔️  New best model saved with Avg Loss: {best_val_loss:.4f}")
        logging.info(f"✔️  New best model saved with Avg Loss: {best_val_loss:.4f}")

print("🎉 Training finished!")
logging.info("Training finished successfully.")
