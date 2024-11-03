import math
import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from torchvision.ops import box_iou

from cc import cc, ccnum
from custom_dataset import CustomDataset

# Run configuration
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"  # make sure this directory exists
model_save_path = os.path.join(OUTPUT_DIR, "inference_graph.pth")  # file name

# Training parameters
NUM_EPOCHS = 20  # Total training cycles
BATCH_SIZE = 16  # Number of images per batch

LEARNING_RATE = 0.001  # Initial step size, usually between 0.0001 and 0.01 ?
STEP_SIZE = 6  # Interval (in epochs) to decay the learning rate, usually 20-30% of the total number of epochs ?
GAMMA = 0.5  # Factor by which the learning rate decays, usually around 0.5 ?

# Creating the model
print(cc("YELLOW", "Creating model..."))
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
)
model.train()
print(cc("GRAY", "Model summary:"))
print(cc("GRAY", str(summary(
    model,
    input_size=(BATCH_SIZE, 3, 512, 512),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))))

# Training parameters
print(cc("CYAN", f"Number of epochs: {NUM_EPOCHS}"))
print(cc("CYAN", f"Batch size: {BATCH_SIZE}"))
print(cc("CYAN", f"Learning rate: {LEARNING_RATE}"))
print(cc("CYAN", f"Step size: {STEP_SIZE}"))
print(cc("CYAN", f"Gamma: {GAMMA}"))
print("-------------------------")

# Device configuration
print(cc("YELLOW", "Configuring devices..."))
CUDA_AVAIL = torch.cuda.is_available()
# CUDA_AVAIL = False  # Force CPU for testing
DEVICE = torch.device("cuda" if CUDA_AVAIL else "cpu")
model.to(DEVICE)
print(cc("BLUE", f"Using device: {DEVICE}"))
print(cc("BLUE", f"CUDA available: {CUDA_AVAIL}"))
if CUDA_AVAIL:
    print(cc("BLUE", f"Number of GPUs: {torch.cuda.device_count()}"))
    print(cc("BLUE", f"GPU: {torch.cuda.get_device_name(0)}"))
print("-------------------------")

# Tracking variables
prev_lr = 0
next_lr = 0
prev_loss = 0
next_loss = 0

# Datasets
print(cc("YELLOW", "Creating datasets..."))
data_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[-20, 20]),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Normalize(mean=[0, 0, 0], std=[0, 0, 0]), # need to compute mean and std
    transforms.ToTensor(),
])
train_dataset = CustomDataset(DATA_DIR, "train", transform=data_transform, device=DEVICE)
test_dataset = CustomDataset(DATA_DIR, "test", device=DEVICE)

print(cc("CYAN", f"Training dataset: {len(train_dataset)} images"))
print(cc("CYAN", f"Batches per epoch: {math.ceil(len(train_dataset) / BATCH_SIZE)}"))
print(cc("CYAN", f"Test dataset: {len(test_dataset)} images"))
print("-------------------------")


def collate_fn(batch):
    return tuple(zip(*batch))


# Data loaders
print(cc("YELLOW", "Creating data loaders..."))
cpu_count = 0
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=cpu_count, collate_fn=collate_fn)

# Model modifications
print(cc("YELLOW", "Preparing model..."))
output_shape = len(train_dataset.label_map) + 1  # add 1 for the background class

# store the original in_features of cls_store layer
in_features = model.roi_heads.box_predictor.cls_score.in_features

# modify cls_store and bbox_pred layers to use output_shape as the out_features parameter
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
    in_features=in_features, out_features=output_shape, bias=True
)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
    in_features=in_features, out_features=output_shape * 4, bias=True
)

# Create the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

input(cc("GREEN", "Ready to begin training with the current configuration. Press any key to continue . . ."))
print("\n")

start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_timer = time.time()
    print(cc("GREEN", f"Beginning epoch {epoch + 1}/{NUM_EPOCHS}..."))
    for images, targets in train_loader:
        images = [image.to(DEVICE) for image in images]  # Move images to the correct device
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # Move targets to the correct device

        y_true = []
        for i in range(len(images)):
            d = {"boxes": targets[i]["boxes"].to(DEVICE), "labels": targets[i]["labels"].to(DEVICE)}
            y_true.append(d)

        losses = model(images, y_true)

        total_loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()  # Clear previous gradients
        total_loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Update learning rate
        # lr_scheduler.step()

        print(cc("BLUE", f"Epoch [{epoch + 1}/{NUM_EPOCHS}]:"))
        print(cc("CYAN",
                 f"Total loss: {total_loss.item()}"
                 f"\nLearning rate: {optimizer.param_groups[0]['lr']}"))
        print(cc("BLUE",
                 f"- Losses:\n"
                 + "\n".join([f"  - {key}: {loss}" for key, loss in losses.items()]) + "\n"))

        # Training loss difference
        next_loss = total_loss.item()
        delta_loss = prev_loss - next_loss
        print(cc("CYAN", f"Training loss delta: {ccnum(delta_loss, reverse=True)}"))
        prev_loss = next_loss

        # Learning rate difference
        next_lr = optimizer.param_groups[0]["lr"]
        delta_lr = prev_lr - next_lr
        print(cc("CYAN", f"Learning rate delta: {ccnum(delta_lr, reverse=True)}"))
        prev_lr = next_lr

    print(cc("GREEN", f"Epoch [{epoch + 1}/{NUM_EPOCHS}] complete! Took {time.time() - epoch_timer:.3f} seconds"))

    # Step the scheduler after each epoch
    lr_scheduler.step()

print(cc("GREEN", f"Training complete! Took {time.time() - start_time:.3f} seconds"))

# Save .pth model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.save(model.state_dict(), model_save_path)
print(cc("GRAY", f"Trained model saved at {model_save_path}"))

input("Press any key to proceed to evaluation . . .")

# Run evaluation on the test dataset
print(cc("GREEN", "Beginning evaluation..."))

model.eval()
total_iou = 0.0
total_images = 0

with torch.no_grad():
    for images, targets in test_loader:
        images = [image.to(DEVICE) for image in images]  # Move images to the correct device
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # Move targets to the correct device
        outputs = model(images)  # Get model predictions

        for i, output in enumerate(outputs):
            # Ground-truth boxes and labels
            gt_boxes = targets[i]["boxes"].to(DEVICE)
            gt_labels = targets[i]["labels"].to(DEVICE)

            # Predicted boxes and labels
            pred_boxes = output["boxes"]
            pred_labels = output["labels"]

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Calculate IoU between each predicted and ground-truth box
                ious = box_iou(pred_boxes, gt_boxes)
                max_iou_per_pred = ious.max(dim=1)[0]  # Take the max IoU for each predicted box

                # Average IoU for the batch
                avg_iou = max_iou_per_pred.mean().item()
                total_iou += avg_iou
                total_images += 1

                # Print some details
                print(f"Image {i + 1}: Avg IoU = {avg_iou:.4f}")
            else:
                print(f"Image {i + 1}: No detections or ground-truth boxes to compare.")

# Calculate and print the overall IoU
if total_images > 0:
    mean_iou = total_iou / total_images
    print(f"\nMean IoU over the test set: {mean_iou:.4f}")
else:
    print("No valid predictions to calculate IoU.")

input("Press any key to continue (exporting to ONNX then to TFLite) . . .")
print("Conversion does not work at the moment")

exit()

# onnx~=1.14.1
# tf2onnx~=1.15.1

# Export the PyTorch model to ONNX format
input_shape = (BATCH_SIZE, 3, 3024, 3024)
dummy_input = torch.randn(input_shape)
onnx_path = os.path.join(OUTPUT_DIR, "saved_model.onnx")
torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=12)

# Convert ONNX model to TFLite
onnx_model = onnx.load(onnx_path)

# Specify target_opset and optimize
tflite_model = onnx_tf.backend.prepare(onnx_model, strict=False)
# https://github.com/onnx/onnx-tensorflow/issues/763
# optimized_onnx_model = tflite_model.graph.as_graph_def()
# tflite_optimized_model = tf2onnx.convert.from_graph_def(optimized_onnx_model, opset=12, output_path=None)

tflite_path = os.path.join(OUTPUT_DIR, "saved_model.tflite")
tflite_model.export_graph(tflite_path)
# with open(tflite_path, "wb") as f:
#     f.write(tflite_model)
