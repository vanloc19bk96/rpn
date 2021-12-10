import torch


def smooth_l1_loss(y_true, y_pred):
    x = torch.abs(y_true - y_pred)
    mask = torch.less(x, 1.0).type(torch.DoubleTensor)
    losses = mask * (0.5 * x ** 2) + (1 - mask) * (x - 0.5)

    return losses


def custom_l1_loss(y_true, y_pred):
    offset_list = y_true[..., :-1]
    label_list = y_true[..., -1]
    y_pred_locs = y_pred[0]
    total_coors, f_w, h_w = list(y_pred_locs.size()[1:])
    y_pred_locs = torch.reshape(y_pred_locs, (-1, (total_coors // 4) * f_w * h_w, 4))
    pos_idx = torch.where(label_list == 1.0)
    pos_idx = torch.stack([pos_idx[0], pos_idx[1]], dim=1)
    pred_boxes = y_pred_locs[pos_idx[:, 0], pos_idx[:, 1], :]
    target_boxes = offset_list[pos_idx[:, 0], pos_idx[:, 1], :]

    # l1_loss = smooth_l1_loss(target_boxes, pred_boxes)
    l1 = torch.nn.SmoothL1Loss()(input=target_boxes, target=pred_boxes)
    return l1


def custom_binary_loss(y_true, y_pred):
    label_list = y_true[..., -1]
    indices = torch.where(label_list != -1)
    indices = torch.stack([indices[0], indices[1]], dim=1)

    y_cls_scores = y_pred[1]
    total_scores, f_w, h_w = list(y_cls_scores.size()[1:])
    y_cls_scores = torch.reshape(y_cls_scores, (-1, total_scores * f_w * h_w))
    pred_score = y_cls_scores[indices[:, 0], indices[:, 1]]
    anchor_score = label_list[indices[:, 0], indices[:, 1]]
    binary_loss = torch.nn.BCELoss()(input=pred_score, target=anchor_score)

    return binary_loss

# a = torch.rand((3, 480, 4))
# label = torch.randint(low=-1, high=2, size=(3, 480, 1))
# a = torch.cat([a, label], -1)
#
# b = torch.rand((3, 4, 4, 30 * 4))
# c = torch.rand((3, 4, 4, 30))
# loss = custom_binary_loss(a, c)
# print(loss)
