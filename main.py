import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

from cc import cc
from custom_dataset import CustomDataset

# Define your data directory
DATA_DIR = "./dataset1"
OUTPUT_DIR = "./output"
model_save_path = os.path.join(OUTPUT_DIR, "inference_graph.pth")
# model_save_path = os.path.join(OUTPUT_DIR, "fasterrcnn_mobilenet_v3_large_fpn.pth")

NUM_EPOCHS = 12
BATCH_SIZE = 8

LEARNING_RATE = 0.002  # usually between 0.0001 and 0.01 ?
STEP_SIZE = 4  # usually 20-30% of the total number of epochs
GAMMA = 0.5  # usually between 0.1 and 0.5 ?
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA is not working for some reason
# DEVICE = torch.device("cpu")

prev_lr = 0
next_lr = 0
prev_loss = 0
next_loss = 0

print(cc("BLUE", f"Using device: {DEVICE}"))

# Define data transforms
data_transform = transforms.Compose([
    # transforms.RandomRotation(degrees=[-20, 20]),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Normalize(mean=[0, 0, 0], std=[0, 0, 0]), # need to compute mean and std
    transforms.ToTensor(),
])

# Datasets
train_dataset = CustomDataset(DATA_DIR, "train", transform=data_transform)
test_dataset = CustomDataset(DATA_DIR, "test", transform=data_transform)

print(cc("BLUE", f"Number of training images: {len(train_dataset)}"))
print(cc("BLUE", f"Number of test images: {len(test_dataset)}"))


def collate_fn(batch):
    return tuple(zip(*batch))


# Data loaders
cpu_count = 0
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cpu_count, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=cpu_count, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
)

# for param in model.parameters():
#     param.requires_grad = False

output_shape = len(train_dataset.label_map)

# store the original in_features of cls_store layer
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Modify cls_store and bbox_pred layers to use output_shape as the out_features parameter
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
    in_features=in_features, out_features=output_shape, bias=True
)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
    in_features=in_features, out_features=output_shape * 4, bias=True
)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

print(cc("BLUE", "Model summary:"))
print(summary(
    model,
    input_size=(BATCH_SIZE, 3, 512, 512),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))
# exit()
print(cc("GREEN", "Beginning training..."))

start_time = time.time()

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_timer = time.time()
    print(cc("GREEN", f"Beginning epoch {epoch + 1}/{NUM_EPOCHS}..."))
    for images, targets in train_loader:
        images = [image.to(DEVICE) for image in images]

        y_true = []
        for i in range(len(images)):
            d = {"boxes": targets[i]["boxes"], "labels": targets[i]["labels"]}
            y_true.append(d)

        losses = model(images, y_true)
        print(losses)

        total_loss = sum(loss for loss in losses.values())

        optimizer.zero_grad()  # Clear previous gradients
        total_loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Update learning rate
        lr_scheduler.step()

        print(cc("BLUE",
                 f"Epoch [{epoch + 1}/{NUM_EPOCHS}]:"
                 f"\n- Total loss: {total_loss.item()}"
                 f"\n- Learning rate: {optimizer.param_groups[0]['lr']}"
                 f"\n- Losses:\n"
                 + "\n".join([f"  - {key}: {loss}" for key, loss in losses.items()])
                 + "\n"))

        # Loss difference
        next_loss = total_loss.item()
        delta_loss = prev_loss - next_loss
        print(cc("CYAN", f"Training loss delta: {delta_loss}"))
        prev_loss = next_loss

        # Learning rate difference
        next_lr = optimizer.param_groups[0]['lr']
        delta_lr = prev_lr - next_lr
        print(cc("CYAN", f"Learning rate delta: {delta_lr}"))
        prev_lr = next_lr

    print(cc("GREEN", f"Epoch [{epoch + 1}/{NUM_EPOCHS}] complete! Took {time.time() - epoch_timer:.3f} seconds"))

print(cc("GREEN", f"Training complete! Took {time.time() - start_time:.3f} seconds"))

# Save .pth model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.save(model.state_dict(), model_save_path)
print(cc("GRAY", f"Trained model saved at {model_save_path}"))

# Evaluate on test set
# cc(""GREEN", "Beginning evaluation...")
# model.eval()  # Set the model to evaluation mode

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
