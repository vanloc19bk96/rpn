import json
import os.path
import xml.etree.ElementTree as ET
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch


def read_annotation(path):
    if not os.path.exists(path):
        raise FileExistsError("Annotations file is not exists!")

    tree = ET.parse(path)
    annotation = tree.getroot()
    bounding_boxes = []
    for boxes in annotation.iter('object'):
        y_min = int(boxes.find("bndbox/ymin").text)
        x_min = int(boxes.find("bndbox/xmin").text)
        y_max = int(boxes.find("bndbox/ymax").text)
        x_max = int(boxes.find("bndbox/xmax").text)
        bounding_boxes.append([x_min, y_min, x_max, y_max])
    return torch.as_tensor(bounding_boxes, dtype=torch.int32)


def write_annotation(image_size, bounding_boxes, saved_path, _filename):
    if not os.path.exists(saved_path):
        raise FileExistsError("Annotations file is not exists!")

    root = ET.Element('annotation')
    folder = ET.SubElement(root, 'folder')
    folder.text = 'images'

    filename = ET.SubElement(root, 'filename')
    filename.text = _filename

    size = ET.SubElement(root, 'size')

    _width, _height = image_size
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    width.text = str(_width)
    height.text = str(_height)
    depth.text = '3'

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    for bounding_box in bounding_boxes:
        object = ET.SubElement(root, 'object')
        name = ET.SubElement(object, 'name')
        name.text = 'trafficlight'
        pose = ET.SubElement(object, 'pose')
        pose.text = '0'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        occluded = ET.SubElement(object, 'occluded')
        occluded.text = '0'
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bounding_box[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bounding_box[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bounding_box[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bounding_box[3])

    tree = ET.ElementTree(root)
    tree.write(os.path.join(saved_path, _filename))


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
