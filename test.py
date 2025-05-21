import torch
from torch.utils.data import DataLoader
from semi_supervised_unet_fixmatch import LabeledMRIDataset, model, DiceLoss

# Paths
test_images = "BraTS_FixMatch/labeled/images"
test_masks = "BraTS_FixMatch/labeled/masks"

# Load full labeled dataset and take 15% for testing
full_dataset = LabeledMRIDataset(test_images, test_masks)
test_size = int(0.15 * len(full_dataset))
_, test_set = torch.utils.data.random_split(full_dataset, [len(full_dataset) - test_size, test_size])
test_loader = DataLoader(test_set, batch_size=4)

# Load best model
model.load_state_dict(torch.load("model_epoch_20.pth"))
model = model.cuda().eval()

# Evaluation
dice_loss = DiceLoss()
total_dice = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.cuda(), masks.cuda()
        outputs = torch.sigmoid(model(images))
        loss = dice_loss(outputs, masks)
        total_dice += (1 - loss.item())

print(f"Test Dice Score: {total_dice / len(test_loader):.4f}")
