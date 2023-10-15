import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from utils import log, Ccodes

from custom_dataset import CustomDataset

# Define your data directory
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"
model_save_path = os.path.join(OUTPUT_DIR, "trained_model.pth")

BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
STEP_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log(f"Using device: {DEVICE}", Ccodes.BLUE)

# Define data transforms
data_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[-45, 45]),  # random rotate
    # transforms.RandomHorizontalFlip(p=0.5),  # random flip
    # transforms.Resize((320, 320), antialias=True),
    transforms.ToTensor(),  # convert to PyTorch tensor
])

# Datasets
train_dataset = CustomDataset(DATA_DIR, "train", transform=data_transform)
test_dataset = CustomDataset(DATA_DIR, "test", transform=data_transform)

log(f"Number of training images: {len(train_dataset)}", Ccodes.BLUE)
log(f"Number of test images: {len(test_dataset)}", Ccodes.BLUE)


def collate_fn(batch):
    return tuple(zip(*batch))


# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
).to(DEVICE)

log("Model summary:", Ccodes.BLUE)
print(summary(
    model,
    input_size=(BATCH_SIZE, 3, 3024, 3024),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds")
))

log("Beginning training...", Ccodes.GREEN)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_timer = time.time()
    log(f"Entering epoch {epoch + 1}/{NUM_EPOCHS}...", Ccodes.GREEN)
    for images, targets in train_loader:
        images = [image.to(DEVICE) for image in images]

        targets_tensor = []
        for i in range(len(images)):
            d = {"boxes": targets[i]["boxes"], "labels": targets[i]["labels"]}
            targets_tensor.append(d)

        loss_dict = model(images, targets_tensor)

        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update learning rate
        lr_scheduler.step()

        log(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]:"
            f"\n- Total loss: {total_loss.item()}"
            f"\n- Learning rate: {optimizer.param_groups[0]['lr']}"
            f"\n- Losses:\n"
            + "\n".join([f"  - {key}: {loss}" for key, loss in loss_dict.items()])
            + "\n",
            Ccodes.BLUE)

    log(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] complete! Took {time.time() - epoch_timer:.3f} seconds", Ccodes.GREEN)

log(f"Training complete! Took {time.time() - start_time:.3f} seconds", Ccodes.GREEN)

# Save model .pth
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.save(model.state_dict(), model_save_path)
log(f"Trained model saved at {model_save_path}", Ccodes.GRAY)

# Evaluate on test set
log("Beginning evaluation...", Ccodes.GREEN)
model.eval()  # Set the model to evaluation mode

total_samples = 0
correct_predictions = 0

for images, targets in test_loader:
    images = [image.to(DEVICE) for image in images]

    with torch.no_grad():
        predictions = model(images)

    for i in range(len(targets)):
        true_labels = targets[i]["labels"]
        pred_labels = predictions[i]["labels"]
        print(true_labels, pred_labels)
        correct_predictions += torch.sum(true_labels == pred_labels).item()
        total_samples += len(true_labels)

accuracy = correct_predictions / total_samples * 100.0

log(f"Test Accuracy: {accuracy:.2f}%", Ccodes.GREEN)
