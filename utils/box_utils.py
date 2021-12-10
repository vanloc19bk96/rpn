import numpy as np
import torch


def to_voc_format(x_center, y_center, width, height):
    x_min = x_center - 0.5 * width
    y_min = y_center - 0.5 * height
    x_max = x_center + 0.5 * width
    y_max = y_center + 0.5 * height

    return [x_min, y_min, x_max, y_max]


def to_center_format(x_min, y_min, x_max, y_max):
    width = x_max - x_min
    height = y_max - y_min

    x_center = x_max - 0.5 * width
    y_center = y_max - 0.5 * height

    return [x_center, y_center, width, height]


def calculate_iou(box1, box2):
    if not torch.is_tensor(box1):
        box1 = torch.as_tensor(box1)
        box2 = torch.as_tensor(box2)

    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])

    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])

    if torch.less(x1, x2) and torch.less(y1, y2):
        w_overlap = x2 - x1
        h_overlap = y2 - y1
        area_overlap = w_overlap * h_overlap
    else:
        return 0
    w_box1 = box1[2] - box1[0]
    h_box1 = box1[3] - box1[1]

    w_box2 = box2[2] - box2[0]
    h_box2 = box2[3] - box2[1]

    area_box1 = w_box1 * h_box1
    area_box2 = w_box2 * h_box2

    area_union_overlap = area_box1 + area_box2
    area_union = area_union_overlap - area_overlap

    return area_overlap / area_union


def resize_bounding_box(bounding_box, image_w, image_h, resized_image_w, resized_image_h):
    x_alter = resized_image_w / image_w
    y_alter = resized_image_h / image_h

    bounding_box[:, 0] = bounding_box[:, 0] * x_alter
    bounding_box[:, 1] = bounding_box[:, 1] * y_alter
    bounding_box[:, 2] = bounding_box[:, 2] * x_alter
    bounding_box[:, 3] = bounding_box[:, 3] * y_alter
    return bounding_box.numpy().astype(int)


def calculate_ious(boxes1, boxes2):
    iou_list = np.zeros((len(boxes1), len(boxes2)))
    for gt_idx, gt_box in enumerate(boxes2):
        for anchor_idx, anchor_box in enumerate(boxes1):
            iou_list[anchor_idx][gt_idx] = calculate_iou(gt_box, anchor_box)

    return iou_list


def calculate_offset(anchors, bounding_boxes, iou_list):
    anchor_center_x, anchor_center_y, anchor_width, anchor_height = to_center_format(
        anchors[:, 0],
        anchors[:, 1],
        anchors[:, 2],
        anchors[:, 3],
    )

    best_ious = np.argmax(iou_list, axis=1)
    gt_coordinates = np.array([bounding_boxes[idx] for idx in best_ious])

    bounding_boxes_center_x, bounding_boxes_center_y, bounding_boxes_center_width, bounding_boxes_center_height = to_center_format(
        gt_coordinates[:, 0],
        gt_coordinates[:, 1],
        gt_coordinates[:, 2],
        gt_coordinates[:, 3]
    )

    eps = np.finfo(anchor_width.dtype).eps
    anchor_width = np.maximum(anchor_width, eps)
    anchor_height = np.maximum(anchor_height, eps)

    dx = (bounding_boxes_center_x - anchor_center_x) / anchor_width
    dy = (bounding_boxes_center_y - anchor_center_y) / anchor_height
    dw = np.log(bounding_boxes_center_width / anchor_width)
    dh = np.log(bounding_boxes_center_height / anchor_height)

    offset_list = np.array([dx, dy, dw, dh]).T
    return offset_list


def offset_to_center(offset, anchors):
    anchor_center_x, anchor_center_y, anchor_width, anchor_height = anchors[:, 0], anchors[:, 1], anchors[:,
                                                                                                  2], anchors[:, 3]
    dx, dy, dw, dh = offset[:, 0], offset[:, 1], offset[:, 2], offset[:, 3]
    center_x = anchor_width * dw + anchor_center_x
    center_y = anchor_height * dh + anchor_center_y
    width = torch.exp(dw) * anchor_width
    height = torch.exp(dh) * anchor_height

    boxes = torch.stack([center_x, center_y, width, height], dim=-1)
    return boxes
