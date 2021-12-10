import argparse
import collections

import torch
import numpy as np

from trainer.trainer import Trainer
from utils.parser import ConfigParser
import data_loaders as module_data
import models as module_model
import losses as module_loss
from utils.utils import prepare_device
import metrics as module_metric

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    rpn_model = config.init_obj('arch', module_model)
    logger.info(rpn_model)

    device, device_ids = prepare_device(config['n_gpu'])
    rpn_model = rpn_model.to(device)

    if len(device_ids) > 1:
        rpn_model = torch.nn.DataParallel(rpn_model, device_ids=device_ids)

    criterions = [getattr(module_loss, loss) for loss in config['losses']]
    metrics = [getattr(module_metric, met['type']) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, rpn_model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    trainer = Trainer(rpn_model, criterions, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

    # import torch.optim as optim
    #
    # optimizer = optim.Adam(rpn_model.parameters(), lr=0.0015)

    # torch.autograd.set_detect_anomaly(True)
    # for epoch in range(100):
    #     rpn_model.train()
    #     running_loss = 0.0
    #     sum_cls_loss = 0.0
    #     batch_size = 0
    #     for i, (data, target) in enumerate(data_loader):
    #         batch_size = data.size(0)
    #         optimizer.zero_grad()
    #         output = rpn_model(data)
    #         loss_locs, loss_cls = [loss(target, output) for loss in criterions]
    #         loss = loss_locs + loss_cls
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         sum_cls_loss += loss_cls.item()
    #     print("Cls loss: ", sum_cls_loss / batch_size)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Object Detection')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loaders;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
