import torch
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
import numpy as np
from utils.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    def __init__(self, model, criterions, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, anchor_list) in enumerate(self.data_loader):
            feature, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(feature)
            loss_locs, loss_cls = [loss(target, output) for loss in self.criterions]
            loss = loss_locs + loss_cls
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for index, met in enumerate(self.metric_ftns):
                self.train_metrics.update(met.__name__, met(output, anchor_list, **self.config['metrics'][index]['args']))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss Locs: {:.6f} Loss Cls: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_locs.item(), loss_cls.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = sum([loss(output, target) for loss in self.criterions])
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = (batch_idx + 1) * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx + 1
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
