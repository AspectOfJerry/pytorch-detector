#main.py
from torchvision.io.image import read_image
from torchvision.models import detection, mobilenetv3
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN
from torchvision.models.resnet import ResNet50_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torch import nn
from torchinfo import summary
from pathlib import Path
import numpy as np
import argparse
import pickle
import torch
import cv2
import ssl
from PIL import Image
from gmodular import data_setup, engine
from timeit import default_timer as timer

#links the unverified context to the default https context, anihilating the need for an effective SSL certificate to download pretrained models
ssl._create_default_https_context = ssl._create_unverified_context

DATA_PATH = Path("data/")
TRAIN_PATH = DATA_PATH / "train"
TEST_PATH = DATA_PATH / "test"
VALIDATION_PATH = DATA_PATH / "validate"
def checkPaths(path):
    if(path.is_dir()):
        print(f"Directory {path} is set for procedure")
    else:
        print(f"Directory {path} is inexistant, making one right now...")
        path.mkdir(parents=True, exist_ok=True)

checkPaths(TRAIN_PATH)
checkPaths(TEST_PATH)
NUM_CLASSES = 1;
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

manual_transforms = transforms.Compose([
    transforms.Resize((640, 640)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])

train_dataset, test_dataset = data_setup.create_YoloDatasets(TRAIN_PATH, TEST_PATH, manual_transforms)
print(train_dataset)
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders_from_datasets(train_set=train_dataset, test_set=test_dataset, batch_size=32) 
#train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(TRAIN_PATH, TEST_PATH, manual_transforms, 32)
#model = createModel(FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, None, None, None, None, None)

#get pretrained weights
weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
#create model from pretrained weights
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)

#freeze all layers
for param in model.parameters():
    param.requires_grad = False
#ready seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#create output_shape from number of labels
output_shape = len(train_dataset.classes)
#store the original in_features of cls_store layer
in_features = model.roi_heads.box_predictor.cls_score.in_features
#change cls_store and bbox_pred layers to use output_shape as the out_features parameter
model.roi_heads.box_predictor.cls_score = torch.nn.Linear(
        in_features=in_features, out_features=output_shape, bias=True
)
model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(
        in_features=in_features, out_features=output_shape * 4, bias=True
)
#print summary
print(output_shape)
print(summary(model=model, 
        input_size=(32, 3, 640, 640), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
))

#train
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = timer()
results = engine.train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_fn,
        5,
        DEVICE)
end_timer = timer()
print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

#prediction

model.eval()

#predict(model, initializeInferenceTransform(FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT), read_image("car.jpg"))
img = read_image("cubesandcones.jpeg")
preprocess = weights.transforms()
batch = [preprocess(img)]
prediction = model(batch)[0]
labels = [weights.meta["categories"][i] for i in prediction["labels"]]
box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                          labels=labels,
                          colors="red",
                          width=4, font_size=30)
im = to_pil_image(box.detach())
im.show()

