import torch
import torchvision
import cv2

color_mapping = {  # BGR format
    "high": (0, 255, 0),  # Green
    "medium": (0, 255, 255),  # Yellow
    "low": (0, 0, 255)  # Red
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
).to(DEVICE)

model.load_state_dict(torch.load("output/trained_model.pth"))
model.eval()

image = cv2.imread("dataset/images/test/3a772b8b-IMG_6417.jpg")

image = image / 255.0  # Normalize the image to values between 0 and 1
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

# Inference
with torch.no_grad():
    output = model(image)

# Extract bboxes, labels, and scores
boxes = output[0]['boxes']  # Assuming the first prediction (index 0)
labels = output[0]['labels']
scores = output[0]['scores']

# Convert image back to NumPy format
image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Resize image to fit within the display window
scale_factor = 1024 / max(image.shape[0], image.shape[1])
image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

# Define label map
label_map = {
    0: "background",
    1: "cube",
    2: "cone",
}

# Draw the bboxes
for box, label, score in zip(boxes, labels, scores):
    box = box.int()
    x, y, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    label_id = int(label)
    label_name = label_map.get(label_id, f'Label {label_id}')  # Get label name from label map or use ID
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
