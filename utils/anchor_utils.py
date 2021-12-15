import numpy as np
import torch

from utils.box_utils import calculate_ious, calculate_offset


def create_anchor_centers(input_size, image_size):
    _, image_width, image_height = image_size
    _, _, input_height, input_width = input_size

    x_stride = image_width / input_width
    y_stride = image_height / input_height
    anchor_x_start_position = x_stride // 2
    anchor_y_start_position = y_stride // 2

    # center of anchor location on image
    x_center = np.arange(anchor_x_start_position, image_width, x_stride)
    y_center = np.arange(anchor_y_start_position, image_width, y_stride)
    centers = np.array(np.meshgrid(x_center, y_center, sparse=False, indexing='xy')).T.reshape(-1, 2)

    return centers, x_stride, y_stride


def generate_anchor(input_size, image_size, bounding_boxes, ratios, scales):
    _, _, input_size_height, input_size_width = input_size

    n_anchors_pos = input_size_height * input_size_width
    # total possible anchors
    n_anchors = n_anchors_pos * len(ratios) * len(scales)
    anchors = np.zeros(shape=(n_anchors, 4))
    center_list, x_stride, y_stride = create_anchor_centers(input_size, image_size)
    count = 0
    for center in center_list:
        center_x, center_y = center
        for ratio in ratios:
            for scale in scales:
                h = np.sqrt(scale ** 2 / ratio)
                w = ratio * h

                # h, w would be really small, scale them with stride
                h *= x_stride
                w *= y_stride
                anchor_xmin = center_x - 0.5 * w
                anchor_ymin = center_y - 0.5 * h
                anchor_xmax = center_x + 0.5 * w
                anchor_ymax = center_y + 0.5 * h
                anchors[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
                count += 1

    valid_anchors, valid_anchor_idx_list = get_valid_anchors(image_size, anchors)
    iou_list = calculate_ious(valid_anchors, bounding_boxes)
    valid_anchor_offset_list = calculate_offset(valid_anchors, bounding_boxes, iou_list)
    offset_list = np.zeros_like(anchors)
    offset_list[valid_anchor_idx_list] = valid_anchor_offset_list
    label_list = np.empty(shape=(anchors.shape[0], 1))
    label_list.fill(-1)
    valid_anchor_label_list = assign_labels(valid_anchors, bounding_boxes)
    label_list[valid_anchor_idx_list] = valid_anchor_label_list
    label_list = sample_anchor_labels(label_list)
    return torch.Tensor(np.concatenate((offset_list, label_list), axis=1)), anchors


def assign_labels(anchors, bounding_boxes):
    label = np.zeros(len(anchors), dtype=np.int)
    label.fill(-1)
    iou_list = calculate_ious(anchors, bounding_boxes)
    best_ious = np.max(iou_list, axis=0)
    top_anchors = np.where(iou_list == best_ious)[0]
    label[top_anchors] = 1
    max_iou = np.max(iou_list, axis=1)
    label[np.where(max_iou > 0.7)[0]] = 1
    label[np.where(max_iou < 0.3)[0]] = 0

    return label.reshape(-1, 1)


def sample_anchor_labels(labels, n_samples=256, neg_ratio=0.5):
    n_foreground = int((1 - neg_ratio) * n_samples)
    n_background = int(neg_ratio * n_samples)

    foreground_index_list = np.where(labels == 1)[0]
    background_index_list = np.where(labels == 0)[0]

    if len(foreground_index_list) > n_foreground:
        ignore_index = foreground_index_list[n_foreground:]
        labels[ignore_index] = -1
    if len(foreground_index_list) < n_foreground:
        diff = n_foreground - len(foreground_index_list)
        n_background += diff
    if len(background_index_list) > n_background:
        ignore_index = background_index_list[n_background:]
        labels[ignore_index] = -1
    return labels


def get_valid_anchors(image_size, anchors):
    _, img_height, img_width = image_size

    inside_anchor_idx_list = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= img_width) &
        (anchors[:, 3] <= img_height)
    )[0]
    inside_anchor_list = anchors[inside_anchor_idx_list]

    return inside_anchor_list, inside_anchor_idx_list
