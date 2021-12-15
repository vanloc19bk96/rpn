import argparse

import torch
from tqdm import tqdm

import data_loaders as module_data
import models as module_arch
from utils.box_utils import to_center_format, offset_to_voc, nms
from utils.parser import ConfigParser


def main(config):
    logger = config.get_logger('inference')
    data_loader = config.init_obj('data_loader', module_data)
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target, anchor_list) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            feature, target = data.to(device), target.to(device)
            pred_anchor_locs, pred_cls_scores = model(feature)
            pred_anchor_locs = pred_anchor_locs.view(data.size(0), 4, -1).transpose(2, 1)
            pred_cls_scores = pred_cls_scores.view(data.size(0), -1)

            roi_min_x, roi_min_y, roi_max_x, roi_max_y = offset_to_voc(pred_anchor_locs, anchor_list)
            roi = torch.stack((roi_min_x, roi_min_y, roi_max_x, roi_max_y)).T
            # clipping the predicted boxes to the image
            roi = torch.clip(roi, 0, config['data_loader']['args']['image_width']).transpose(1, 0)
            min_size = 16
            width = roi[..., 2] - roi[..., 0]  # xmax - xmin
            height = roi[..., 3] - roi[..., 1]  # ymin - ymax
            keep = torch.where((width > min_size) & (height > min_size))
            keep = torch.stack((keep[0], keep[1]), dim=-1)
            roi = roi[keep[:, 0], keep[:, 1]].view(data.size(0), -1, 4)
            score = pred_cls_scores[keep[:, 0], keep[:, 1]].view(data.size(0), -1)
            sorted_idx = torch.argsort(score, dim=1, descending=True)
            score_sorted = torch.squeeze(score[:, sorted_idx], dim=1)
            roi_sorted = torch.squeeze(roi[:, sorted_idx], dim=1)
            nms_top = 12000
            score_sorted = score_sorted[:, :nms_top]
            roi_sorted = roi_sorted[:, :nms_top]

            n_train_post_nms = 2000
            keep = nms(roi_sorted, nms_top)
            keep = keep[:, :n_train_post_nms]
            roi_sorted = torch.squeeze(roi_sorted[:, keep], dim=1)
            score_sorted = torch.squeeze(score_sorted[:, keep], dim=1)
            return roi_sorted, score_sorted


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args, training=False)
    main(config)
