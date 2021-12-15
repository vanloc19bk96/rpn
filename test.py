import argparse

import torch
from tqdm import tqdm

from utils.parser import ConfigParser
import data_loaders as module_data
import models as module_arch
import losses as module_loss
import metrics as module_metric


def main(config):
    logger = config.get_logger('test')
    data_loader = config.init_obj('data_loader', module_data)
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    criterions = [getattr(module_loss, loss) for loss in config['losses']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    metrics = [getattr(module_metric, met['type']) for met in config['metrics']]
    total_metrics = torch.zeros(len(metrics))
    total_loss = 0.0
    with torch.no_grad():
        for i, (data, target, anchor_list) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            feature, target = data.to(device), target.to(device)
            output = model(feature)
            loss_locs, loss_cls = [loss(target, output) for loss in criterions]
            loss = loss_locs + loss_cls

            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)

    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


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
