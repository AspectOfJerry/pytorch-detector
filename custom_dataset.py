import os
import xml

import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import log, Ccodes


# Modify the CustomDataset class to handle images with no annotations
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_split, transform=None):
        self.root_dir = root_dir
        self.data_split = data_split  # "train" and "test"
        self.image_dir = os.path.join(root_dir, "images", data_split)
        self.annotation_dir = os.path.join(root_dir, "annotations", data_split)
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform

        self.label_map = {
            "0": 0
        }

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_file).convert("RGB")

        # Load and parse the XML annotation file
        xml_file = os.path.join(self.annotation_dir, f"{os.path.splitext(self.image_files[idx])[0]}.xml")
        bounding_boxes = self.parse_xml_annotation(xml_file)

        # Create a list of bounding boxes
        target_boxes = torch.tensor([bb["boxes"] for bb in bounding_boxes], dtype=torch.float32)

        # Convert the list of labels to a list of class indices
        labels = [label for bb in bounding_boxes for label in bb["labels"]]
        label_indices = torch.tensor([self.label_map[label] for label in labels], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        targets = {
            "boxes": target_boxes,
            "labels": label_indices,
        }

        return image, targets

    def __len__(self):
        return len(self.image_files)

    def parse_xml_annotation(self, xml_file):
        log(f"Parsing {xml_file}")
        tree = xml.etree.ElementTree.parse(xml_file)
        root = tree.getroot()

        bounding_boxes = []
        for obj in root.findall("object"):
            label = str(obj.find("name").text)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bounding_boxes.append({"labels": [label], "boxes": [xmin, ymin, xmax, ymax]})

        log(f"- Bounding boxes: {bounding_boxes}", Ccodes.GRAY)
        return bounding_boxes
