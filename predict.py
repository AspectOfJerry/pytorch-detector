import torch
import torchvision
import cv2

color_mapping = {
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

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
)
in_features = model.roi_heads.box_predictor.cls_score.in_features
output_shape = len(label_map)

model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
    in_features=in_features, out_features=output_shape, bias=True
)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
    in_features=in_features, out_features=output_shape * 4, bias=True
)

model.load_state_dict(torch.load("output/fasterrcnn_mobilenet_v3_large_320_fpn.pth"))
model.eval()

# change the id if needed (multiple cameras, 0 default)
cap = cv2.VideoCapture(1)

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

    cv2.imshow("Object Detection [cone, cube]", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
