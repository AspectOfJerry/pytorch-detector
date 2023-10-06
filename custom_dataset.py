import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

label_to_index_mapping = {
    "cone": 0,
    "cube": 1,
}


class CustomDataset(Dataset):

    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split  # "train" or "test", specify the split
        self.transform = transform

        # Define the image and annotation directories for train and test splits
        self.image_dir = os.path.join(data_dir, "images", split)
        self.annotation_dir = os.path.join(data_dir, "annotations", split)

        # Get a list of image file names
        self.image_files = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_files)

    def parse_xml_annotation(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bounding_boxes = []

        for obj in root.findall("object"):
            label = obj.find("name").text  # Extract the label as a string

            bbox = obj.find("bndbox")
            x1 = int(bbox.find("xmin").text)
            y1 = int(bbox.find("ymin").text)
            x2 = int(bbox.find("xmax").text)
            y2 = int(bbox.find("ymax").text)

            bounding_boxes.append({
                "boxes": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
                "labels": [label]  # Keep label as a list of strings
            })

        return bounding_boxes

    def __getitem__(self, idx):
        # Load an image
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_file).convert("RGB")

        # Load and parse the corresponding XML annotation file
        xml_file = os.path.splitext(image_file)[0] + ".xml"
        xml_file = os.path.join(self.annotation_dir, xml_file)
        bounding_boxes = self.parse_xml_annotation(xml_file)

        target = {}
        target["boxes"] = torch.stack([bb["boxes"] for bb in bounding_boxes])

        # Convert the list of labels to a list of class indices
        labels = [label for bb in bounding_boxes for label in bb["labels"]]
        label_indices = [label_to_index_mapping[label] for label in labels]

        target["labels"] = torch.tensor(label_indices, dtype=torch.int64)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
