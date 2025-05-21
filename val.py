import torch
from torch.utils.data import DataLoader
from dataloader import LabeledDataset
import torch.nn as nn
import os
import logging
import numpy as np

# Configuration
BASE_PATH = './BraTS_FixMatch'
TEST_IMAGES_DIR = os.path.join(BASE_PATH, 'labeled', 'images')
TEST_MASKS_DIR = os.path.join(BASE_PATH, 'labeled', 'masks')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'fixmatch_model.pth')
BATCH_SIZE = 4
MODALITIES = ('t1', 't1ce', 't2', 'flair')
LOG_FILE = os.path.join(BASE_PATH, 'evaluation.log')

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting model evaluation...")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Function to calculate Dice score
def calculate_dice(outputs, targets, smooth=1e-7):
    """Calculates the Dice score.
    Args:
        outputs (torch.Tensor): The model's predicted probabilities.
        targets (torch.Tensor): The ground truth labels.
        smooth (float): Smoothing factor to avoid division by zero.
    Returns:
        float: The Dice score.
    """
    outputs = outputs.squeeze(1)
    targets = targets.squeeze(1)
    intersection = (outputs * targets).sum(dim=1).sum(dim=1)
    union = outputs.sum(dim=1).sum(dim=1) + targets.sum(dim=1).sum(dim=1)
    dice = ((2 * intersection + smooth) / (union + smooth)).mean()
    return dice

# Prepare Dataset and DataLoader
logging.info("Preparing test dataset and dataloader...")
test_dataset = LabeledDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR, modalities=MODALITIES)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
logging.info(f"Test set size: {len(test_dataset)}")

# Model Setup
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model = model.to(device).eval()
logging.info(f"Loaded model from: {CHECKPOINT_PATH}")
logging.info(f"Model architecture: {model}")

# Evaluation
logging.info("Starting evaluation...")
dice_loss = nn.BCEWithLogitsLoss() # change to the loss used during training
total_dice = 0
total_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = dice_loss(outputs, masks.float()) # ensure correct type
        probabilities = torch.sigmoid(outputs)
        dice_score = calculate_dice(probabilities, masks)
        total_dice += dice_score.item()
        total_loss += loss.item()

avg_dice = total_dice / len(test_loader)
avg_loss = total_loss / len(test_loader)
logging.info(f"Average Test Dice Score: {avg_dice:.4f}")
logging.info(f"Average Test Loss: {avg_loss:.4f}")
print(f"Test Dice Score: {avg_dice:.4f}")
print(f"Test Loss: {avg_loss:.4f}") # print also the loss

logging.info("Evaluation finished!")
