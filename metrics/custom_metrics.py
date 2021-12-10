from utils.box_utils import to_voc_format, calculate_iou, offset_to_center, to_center_format
import torch
from sklearn.metrics import precision_score, recall_score
import numpy as np

def average_precision(y_pred, y_true, **kwargs):
    low_threshold = kwargs['low_threshold']
    high_threshold = kwargs['high_threshold']
    num_threshold = kwargs['num_threshold']

    anchor_list = y_true[..., :-1]
    label_list = y_true[..., -1]

    pos_idx = torch.where(label_list != -1)
    indices = torch.stack([pos_idx[0], pos_idx[1]], dim=1)

    y_pred_locs = y_pred[0]
    total_coors, f_w, h_w = list(y_pred_locs.size()[1:])
    y_pred_locs = torch.reshape(y_pred_locs, (-1, (total_coors // 4) * f_w * h_w, 4))
    y_pred_locs = y_pred_locs[indices[:, 0], indices[:, 1]]
    anchor_list = anchor_list[indices[:, 0], indices[:, 1]]
    label_list = label_list[indices[:, 0], indices[:, 1]]

    anchor_center_x, anchor_center_y, anchor_width, anchor_height = to_center_format(
        anchor_list[:, 0],
        anchor_list[:, 1],
        anchor_list[:, 2],
        anchor_list[:, 3],
    )

    anchor_list = torch.stack((anchor_center_x, anchor_center_y, anchor_width, anchor_height), dim=-1)

    y_pred_locs = offset_to_center(y_pred_locs, anchor_list)

    x_true_min, y_true_min, x_true_max, y_true_max = to_voc_format(
        anchor_list[:, 0],
        anchor_list[:, 1],
        anchor_list[:, 2],
        anchor_list[:, 3])

    x_pred_min, y_pred_min, x_pred_max, y_pred_max = to_voc_format(
        y_pred_locs[:, 0],
        y_pred_locs[:, 1],
        y_pred_locs[:, 2],
        y_pred_locs[:, 3])

    boxes1 = torch.stack((x_true_min, y_true_min, x_true_max, y_true_max), dim=-1)
    boxes2 = torch.stack((x_pred_min, y_pred_min, x_pred_max, y_pred_max), dim=-1)

    thresholds = torch.linspace(low_threshold, high_threshold, num_threshold)
    precisions = []
    recalls = []

    y_true = np.where(label_list.numpy() == 1, "positive", "negative")
    for threshold in thresholds:
        ious = torch.tensor([calculate_iou(torch.flatten(box1), torch.flatten(boxes2[index]))
                             for index, box1 in enumerate(boxes1)])
        y_pred = np.where(ious >= threshold, "positive", "negative")
        precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")

        precisions.append(precision)
        recalls.append(recall)

    AP = np.sum((np.array(recalls)[:-1] - np.array(recalls)[1:]) * np.array(precisions[:-1]))
    return AP / 2
