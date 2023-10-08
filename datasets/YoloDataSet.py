from torchvision import transforms
from typing import Dict, List
from PIL import Image
import os
import torch

class YoloDataSet:
    def __init__(self, path: str, transform: transforms):
        self.root_dir = path
        self.labelMap = self.decodeFromAnnotationFile()
        self.image_files = os.listdir(path)
        self.transform = transform
        self.classes = self.decodeFromClassesFile()

    def __len__(self):
        return len(self.image_files)

        
    def __getitem__(self, idx):
        image = Image.open(self.root_dir / self.image_files[idx]).convert("RGB")
        target_boxes = [torch.tensor(self.labelMap[instance]["box"], dtype=torch.int32) for instance in self.labelMap]
        target_labels = [self.labelMap[instance]["label"] for instance in self.labelMap]
        target = {
                "boxes" : target_boxes,
                "labels" : torch.tensor(target_labels, dtype=torch.int32)
            }
        if(self.transform):
            print("TRANSFORM")
            image = self.transform(image)
        return tuple((image, target))


    def decodeFromAnnotationFile(self) -> Dict[str, Dict[str, List]]:
        boxes = {}
        with open(self.root_dir / "_annotations.txt") as f:
            line = f.readline()
            while line:
                elements = line.split(" ")
                if(len(elements) < 1): raise ValueError("The line : {line}; does not contain any information")
                filename = elements[0]
                for i in range(len(elements) - 1):
                    datas = elements[i+1].split(",")
                    x = int(datas[0])
                    y = int(datas[1])
                    aX = int(datas[2])
                    aY = int(datas[3])
                    label = int(datas[4])
                    print(f"{filename} with box starting at ({x}, {y}) and ending at ({aX}, {aY}) with label {label}")
                    boxes[filename] = {"box" : [x, y, aX, aY], "label" : [label]}
                line = f.readline()
        return boxes
    
    def decodeFromClassesFile(self) -> List :
        classes = []
        with open(self.root_dir / "_classes.txt") as f:
            line = f.readline()
            while line:
                classes.append(line)
                line = f.readline()
        return classes

        
