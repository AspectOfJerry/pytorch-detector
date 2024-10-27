import time

import torch
import torchvision
import cv2
from torchinfo import summary

from cc import cc

color_mapping = {
    "high": (0, 255, 0),  # Green
    "medium": (0, 255, 255),  # Yellow
    "low": (0, 0, 255)  # Red
}

label_map = {
    0: "note"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA is not working for some reason
# DEVICE = torch.device("cpu")

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
)

# store the original in_features of cls_store layer
in_features = model.roi_heads.box_predictor.cls_score.in_features
output_shape = len(label_map)
print(output_shape)

model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
    in_features=in_features, out_features=output_shape, bias=True
)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
    in_features=in_features, out_features=output_shape * 4, bias=True
)

model.load_state_dict(torch.load("output/inference_graph.pth"))
model.eval()

print(cc("BLUE", "Model summary:"))
print(summary(
    model,
    input_size=(1, 3, 512, 512),
    verbose=0,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    row_settings=["var_names"]
))

# Read and preprocess the image
# image = cv2.imread('dataset1/images/test/000000000116.jpeg')
image = cv2.imread('dataset1/images/train/000000000001.jpeg')
# image = cv2.resize(image, (512, 512))  # Resize to match the input size
image = image / 255.0  # Normalize the image to values between 0 and 1
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    output = model(image)

# Extract bboxes, labels, and scores
boxes = output[0]['boxes']
labels = output[0]['labels']
scores = output[0]['scores']

# Convert image back to NumPy format
image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
scale_factor = 512 / max(image.shape[0], image.shape[1])

# Draw the bboxes
for box, label, score in zip(boxes, labels, scores):
    box = box.int()

    # Scale the bounding box coordinates because the image was resized
    x, y, x_max, y_max = int(box[0] * scale_factor), int(box[1] * scale_factor), int(box[2] * scale_factor), int(box[3] * scale_factor)
    label_id = int(label)
    label_name = label_map.get(label_id, f'Label {label_id}')
    score = round(score.item(), 2)

    if score >= 0.8:
        color = color_mapping['high']
    elif score >= 0.5:
        color = color_mapping['medium']
    else:
        color = color_mapping['low']

    cv2.rectangle(image, (x, y), (x_max, y_max), color, 2)
    cv2.putText(image, f'Label: {label_name}, Score: {score}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow('Image with Predictions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

# change the id if needed (multiple cameras, 0 default)
cap = cv2.VideoCapture(0)

previousT = 0
currentT = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    model_frame = frame.copy()

    # Resize image to fit within the display window
    scaleFactorX = 640 / frame.shape[1]
    scaleFactorY = 480 / frame.shape[0]

    frame = cv2.resize(frame, (0, 0), fx=scaleFactorX, fy=scaleFactorY)

    # Normalize
    model_frame = frame / 255.0
    model_frame = torch.tensor(model_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(model_frame)

    boxes = output[0]["boxes"]
    labels = output[0]["labels"]
    scores = output[0]["scores"]

    print(scores)

    for box, label, score in zip(boxes, labels, scores):
        box = box.int()
        x, y, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Scale the bounding box coordinates
        x = int(x * scaleFactorX)
        y = int(y * scaleFactorY)
        x_max = int(x_max * scaleFactorX)
        y_max = int(y_max * scaleFactorY)

        label_id = int(label)
        label_name = label_map.get(label_id, f"Label {label_id}")
        score = round(score.item(), 2)

        if score <= 0.4:
            continue
        print(score)

        if score >= 0.8:
            color = color_mapping["high"]
            cv2.rectangle(frame, (x, y), (x_max, y_max), color, 3)
            cv2.putText(frame, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif score >= 0.5:
            color = color_mapping["medium"]
            cv2.rectangle(frame, (x, y), (x_max, y_max), color, 2)
            cv2.putText(frame, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # else:
        #     color = color_mapping["low"]
        #     cv2.rectangle(frame, (x, y), (x_max, y_max), color, 1)

    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT

    cv2.putText(frame, str(round(fps, 4)) + " fps", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection [cone, cube]", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
