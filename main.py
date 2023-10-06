import os

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Define the custom dataset
from custom_dataset import CustomDataset
from utils import log, Ccodes

# Define your data directory
DATA_DIR = "./dataset"  # Replace with your data directory
OUTPUT_DIR = "./output"  # Replace with your output directory
model_save_path = os.path.join(OUTPUT_DIR, "trained_model.pth")

NUM_CLASSES = 3  # Number of classes **including background (+1)**
BATCH_SIZE = 4
NUM_EPOCHS = 12
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms
data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=[-45, 45]),  # random rotate
    transforms.RandomHorizontalFlip(p=0.5),  # random flip
    transforms.ToTensor()  # convert to PyTorch tensor
])

# Create training and validation datasets
train_dataset = CustomDataset(DATA_DIR, "train", transform=data_transform)
test_dataset = CustomDataset(DATA_DIR, "test", transform=data_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Define the Faster R-CNN model with a MobileNetV3 backbone
backbone = torchvision.models.mobilenet_v3_small(pretrained=True)
backbone.out_channels = 960  # Output channels of the backbone
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

model = FasterRCNN(
    backbone,
    num_classes=NUM_CLASSES,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
).to(DEVICE)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# Define the evaluation function
def evaluate_model(_model, _test_loader, device):
    _model.eval()
    coco_evaluator = torchvision.ops.CocoEvaluator(
        torchvision.datasets.CocoDetection(DATA_DIR, "test", transform=None),
        iou_types=["bbox"],
    )

    with torch.no_grad():
        for _images, _targets in _test_loader:
            _images = list(image.to(device) for image in _images)
            _targets = [{k: v.to(device) for k, v in t.items()} for t in _targets]

            outputs = _model(_images)

            # Calculate evaluation metrics
            for target, output in zip(_targets, outputs):
                coco_evaluator.update(target, output)

    # Compute and print evaluation metrics
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    log("Evaluation complete.", Ccodes.GREEN)


# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    log(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {losses.item()}")
    log(f"Learning rate: {optimizer.param_groups[0]['lr']}\n")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
log(f"Trained model saved at {model_save_path}", Ccodes.GREEN)

# Comment this line if you don't want to evaluate the model on the test set.
evaluate_model(model, test_loader, DEVICE)
