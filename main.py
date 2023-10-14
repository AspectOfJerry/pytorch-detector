import os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.rpn import AnchorGenerator
from utils import log, Ccodes

from custom_dataset import CustomDataset

# Define your data directory
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"
model_save_path = os.path.join(OUTPUT_DIR, "trained_model.pth")

NUM_CLASSES = 3  # number of classes **includes background (+1)**
BATCH_SIZE = 2
NUM_EPOCHS = 12
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log(f"Using device: {DEVICE}", Ccodes.BLUE)

# Define data transforms
data_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[-45, 45]),  # random rotate
    # transforms.RandomHorizontalFlip(p=0.5),  # random flip
    # transforms.Resize((756, 756), antialias=True),  # resize the image to (756, 756) 3024/4
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

# Faster R-CNN with a MobileNetV3 backbone
backbone = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
backbone.out_channels = 960  # output channels of the backbone
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

model = torchvision.models.detection.FasterRCNN(
    backbone,
    num_classes=NUM_CLASSES,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
).to(DEVICE)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, targets in train_loader:
        images = list(image for image in images)
        # images = torch.stack([image.to(DEVICE) for image in images])

        [log(image.shape, Ccodes.GRAY) for image in images]
        log(f"Targets: {targets}", Ccodes.GRAY)
        log(f"Targets type: {type(targets)}", Ccodes.GRAY)

        # targets_tensor = [{key: value.clone().detach().to(DEVICE) for key, value in target.items()} for target in targets]
        # targets_tensor = [{key: torch.tensor(value) for key, value in t.items()} for t in targets[0]]

        targets_tensor = []
        for i in range(len(images)):
            d = {}
            d["boxes"] = targets[i]["boxes"]
            d["labels"] = targets[i]["labels"]
            targets_tensor.append(d)

        log(f"Targets tensor: {targets_tensor}", Ccodes.GRAY)
        [log(f"Targets tensor shape: {box['boxes'].shape}", Ccodes.GRAY) for box in targets_tensor]

        # targets_tensor = [{key: value for key, value in t.items()} for t in targets_tensor]

        loss_dict = model(images, targets_tensor)

        for key, loss in loss_dict.items():
            print(f"{key}: {loss}")
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update learning rate
        lr_scheduler.step()

        log(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {loss_dict['loss'].item()}")
        log(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")

# Save model .pth
torch.save(model.state_dict(), model_save_path)
log(f"Trained model saved at {model_save_path}", Ccodes.GREEN)
