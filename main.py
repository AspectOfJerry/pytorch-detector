import os
import time

import onnx
import onnx_tf
import tf2onnx
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary

from custom_dataset import CustomDataset
from utils import log, Ccodes

# Define your data directory
DATA_DIR = "./dataset"
OUTPUT_DIR = "./output"
model_save_path = os.path.join(OUTPUT_DIR, "fasterrcnn_mobilenet_v3_large_fpn.pth")

BATCH_SIZE = 8
NUM_EPOCHS = 16
LEARNING_RATE = 0.001
STEP_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log(f"Using device: {DEVICE}", Ccodes.BLUE)

# Define data transforms
data_transform = transforms.Compose([
    transforms.RandomRotation(degrees=[-20, 20]),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Normalize(mean=[0, 0, 0], std=[0, 0, 0]), # need to compute mean and std
    # transforms.Resize((320, 320), antialias=True),
    transforms.ToTensor(),
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

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = False

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
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.9)

log("Model summary:", Ccodes.BLUE)
print(summary(
    model,
    input_size=(BATCH_SIZE, 3, 3024, 3024),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))

log("Beginning training...", Ccodes.GREEN)

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

# Save .pth model
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.save(model.state_dict(), model_save_path)
log(f"Trained model saved at {model_save_path}", Ccodes.GRAY)

# Evaluate on test set
# log("Beginning evaluation...", Ccodes.GREEN)
# model.eval()  # Set the model to evaluation mode

input("Press any key to continue (exporting to ONNX then to TFLite) . . .")
print("Conversion does not work at the moment")
exit()

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
