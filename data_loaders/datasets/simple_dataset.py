import os

from torch.utils.data import Dataset
from pathlib import Path
from cv2 import cv2
from utils.anchor_utils import generate_anchor
from utils.utils import read_annotation
from utils.box_utils import resize_bounding_box
import torch
import models.backbone as module_backbone


class SimpleDataset(Dataset):
    def __init__(self, training_data_dir, testing_data_dir, image_width, image_height, scales=None, ratios=None,
                 transformation=None, generate_input_model=None, training=True):
        if ratios is None:
            ratios = [0.5, 1, 2]
        if scales is None:
            scales = [8, 16, 32]

        if training:
            self.data_dir = training_data_dir
        else:
            self.data_dir = testing_data_dir
        self.image_width = image_width
        self.image_height = image_height
        self.scales = scales
        self.ratios = ratios

        self.image_paths = list(os.listdir(os.path.join(self.data_dir, 'images')))
        self.annotation_paths = list(os.listdir(os.path.join(self.data_dir, 'annotations')))
        self.transformation = transformation
        self.generate_input_model = generate_input_model

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, "images", self.image_paths[index])
        annotation_path = list(filter(lambda x: Path(x).stem == Path(img_path).stem,
                                      self.annotation_paths))[0]
        annotation_path = os.path.join(self.data_dir, "annotations", annotation_path)

        image = cv2.imread(img_path)
        bounding_boxes = read_annotation(annotation_path)
        if self.transformation is not None:
            args = {'image_size': (self.image_width, self.image_height), 'image': image, 'boxes': bounding_boxes}
            image, bounding_boxes = self.transformation(args)
        else:
            before_resize_height, before_resize_width = image.shape[:-1]
            image = cv2.resize(image, (self.image_width, self.image_height)) / 255.0
            bounding_boxes = resize_bounding_box(bounding_boxes, before_resize_width, before_resize_height,
                                                 self.image_width,
                                                 self.image_height)
        generate_input_model = getattr(module_backbone, self.generate_input_model)()

        image = image / 255.0
        image = torch.transpose(torch.tensor(image.copy()), 2, 0).type(torch.FloatTensor)
        input_feature = generate_input_model(torch.unsqueeze(image, 0))

        offset_label_list, anchor_list = generate_anchor(input_feature.size(), image.size(), bounding_boxes,
                                                         self.ratios,
                                                         self.scales)
        return torch.squeeze(input_feature, 0), offset_label_list, anchor_list
