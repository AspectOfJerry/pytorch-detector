import torch
import torchvision
import cv2

color_mapping = {  # BGR format
    "high": (0, 255, 0),  # Green
    "medium": (0, 255, 255),  # Yellow
    "low": (0, 0, 255)  # Red
}

label_map = {
    0: "background",
    1: "cube",
    2: "cone",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
# weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
# ).to(DEVICE)

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
)

# store the original in_features of cls_store layer
in_features = model.roi_heads.box_predictor.cls_score.in_features

output_shape = len(label_map)

# Modify cls_store and bbox_pred layers to use output_shape as the out_features parameter
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
    in_features=in_features, out_features=output_shape, bias=True
)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
    in_features=in_features, out_features=output_shape * 4, bias=True
)

model.load_state_dict(torch.load("output/trained_model.pth"))
model.eval()

image = cv2.imread("dataset/images/train/bae58b57-IMG_6399.jpg")

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

scale_factor = 1024 / max(image.shape[0], image.shape[1])

# Resize image to fit within the display window
image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

# Draw the bboxes
for box, label, score in zip(boxes, labels, scores):
    box = box.int()
    # Scale the bounding box coordinates because the image was resized
    x, y, x_max, y_max = int(box[0] * scale_factor), int(box[1] * scale_factor), int(box[2] * scale_factor), int(box[3] * scale_factor)

    label_id = int(label)
    label_name = label_map.get(label_id, f'Label {label_id}')
    score = round(score.item(), 2)

    if score >= 0.8:
        color = color_mapping["high"]
    elif score >= 0.5:
        color = color_mapping["medium"]
    else:
        color = color_mapping["low"]

    cv2.rectangle(image, (x, y), (x_max, y_max), color, 2)
    cv2.putText(image, f"Label: {label_name}, Score: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow("Image with Predictions", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
