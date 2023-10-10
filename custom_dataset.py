import os
import xml.etree.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import log, Ccodes


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_split, transform=None):
        self.root_dir = root_dir
        self.data_split = data_split  # "train" or "test"
        self.image_dir = os.path.join(root_dir, "images", data_split)
        self.annotation_dir = os.path.join(root_dir, "annotations", data_split)
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform
        self.label_to_index_mapping = {"cube": 0, "cone": 1}

    def __getitem__(self, idx):
        # Load an image
        image_file = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_file).convert("RGB")

        # Load and parse the corresponding XML annotation file
        xml_file = os.path.join(self.annotation_dir, f"{os.path.splitext(self.image_files[idx])[0]}.xml")
        bounding_boxes = self.parse_xml_annotation(xml_file)

        # Create a list of bounding boxes for this image
        target_boxes = [torch.tensor(bb["boxes"], dtype=torch.float32) for bb in bounding_boxes]

        # Convert the list of labels to a list of class indices
        labels = [label for bb in bounding_boxes for label in bb["labels"]]
        label_indices = [self.label_to_index_mapping[label] for label in labels]

        targets = {
            "boxes": target_boxes,
            "labels": label_indices,
            "num_boxes": len(target_boxes)
        }

        if self.transform:
            image = self.transform(image)  # Pass only the image to the data transformation object
            targets = {
                "boxes": target_boxes,
                "labels": label_indices,
                "num_boxes": len(target_boxes)
            }

        return image, targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Pad the target tensors to have the same size
        max_num_boxes = max(target["num_boxes"] for target in targets)
        padded_targets = {
            "boxes": torch.zeros((len(targets), max_num_boxes, 4), dtype=torch.float32),
            "labels": torch.zeros((len(targets), max_num_boxes), dtype=torch.int64),
            "num_boxes": torch.zeros(len(targets), dtype=torch.int64)
        }

        for i, target in enumerate(targets):
            num_boxes = target["num_boxes"]
            padded_targets["boxes"][i, :num_boxes] = target["boxes"]
            padded_targets["labels"][i, :num_boxes] = target["labels"]
            padded_targets["num_boxes"][i] = num_boxes

        return torch.stack(images), padded_targets

    def parse_xml_annotation(self, xml_file):
        log(f"Parsing {xml_file}")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bounding_boxes = []
        for obj in root.findall("object"):
            label = str(obj.find("name").text)  # Ensure label is a string
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bounding_boxes.append({"labels": [label], "boxes": [xmin, ymin, xmax, ymax]})

        log(f"- Bounding boxes: {bounding_boxes}", Ccodes.GRAY)
        return bounding_boxes
