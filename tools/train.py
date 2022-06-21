# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2022/3/28 17:51
# @Software: PyCharm
# @Brief: train script
from args import args, dev, class_names

from core.loss import focal_loss, l1_loss
from core.helper import remove_dir_and_create_dir, get_model, get_dataset, draw_bbox
from core.detect import postprocess_output, decode_bbox
from core.dataset import recover_input
from net import CenterNetPoolingNMS

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import os
import math


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult    # cycle steps magnification
        self.base_max_lr = max_lr   # first max learning rate
        self.max_lr = max_lr    # max learning rate in the current cycle
        self.min_lr = min_lr    # min learning rate
        self.warmup_steps = warmup_steps    # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps    # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch     # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def get_summary_image(images,
                      hms_true, whs_true, offsets_true,
                      hms_pred, whs_pred, offsets_pred, dev):

    summary_images = []

    outputs_true = postprocess_output(hms_true, whs_true, offsets_true, args.confidence, dev)
    outputs_pred = postprocess_output(hms_pred, whs_pred, offsets_pred, args.confidence, dev)
    outputs_true = decode_bbox(outputs_true,
                               (args.input_height, args.input_height),
                               dev, need_nms=True, nms_thres=0.4)
    outputs_pred = decode_bbox(outputs_pred,
                               (args.input_height, args.input_height),
                               dev, need_nms=True, nms_thres=0.4)

    images = images.cpu().numpy()
    for i in range(len(images)):
        image = images[i]
        image = recover_input(image.copy())

        output_true = outputs_true[i]
        output_pred = outputs_pred[i]

        if len(output_true) != 0:
            output_true = output_true.data.cpu().numpy()
            labels_true = output_true[:, 5]
            bboxes_true = output_true[:, :4]
        else:
            labels_true = []
            bboxes_true = []

        if len(output_pred) != 0:
            output_pred = output_pred.data.cpu().numpy()
            labels_pred = output_pred[:, 5]
            bboxes_pred = output_pred[:, :4]
        else:
            labels_pred = []
            bboxes_pred = []

        image_true = draw_bbox(image, bboxes_true, labels_true, class_names)
        image_pred = draw_bbox(image, bboxes_pred, labels_pred, class_names)

        summary_images.append(np.hstack((image_true, image_pred)).astype(np.uint8))

    return summary_images


def train_one_epochs(model, train_loader, epoch, optimizer, scheduler, dev, writer):
    global step

    model.train()
    tbar = tqdm(train_loader)

    total_loss = []
    image_write_step = len(train_loader)

    for images, hms_true, whs_true, offsets_true, offset_masks_true in tbar:
        tbar.set_description("epoch {}".format(epoch))

        # Set variables for training
        images = images.float().to(dev)
        hms_true = hms_true.float().to(dev)
        whs_true = whs_true.float().to(dev)
        offsets_true = offsets_true.float().to(dev)
        offset_masks_true = offset_masks_true.float().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss
        training_output = model(images, mode='train', ground_truth_data=(hms_true,
                                                                         whs_true,
                                                                         offsets_true,
                                                                         offset_masks_true))
        hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true = training_output

        loss = loss.mean()
        c_loss = c_loss.mean()
        wh_loss = wh_loss.mean()
        off_loss = off_loss.mean()

        total_loss.append(loss.item())

        if step % image_write_step == 0:
            summary_images = get_summary_image(images,
                                               hms_true, whs_true, offsets_true,
                                               hms_pred, whs_pred, offsets_pred, dev)
            for i, summary_image in enumerate(summary_images):
                writer.add_image('train_images_{}'.format(i), summary_image, global_step=step, dataformats="HWC")

        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("c_loss", c_loss.item(), step)
        writer.add_scalar("wh_loss", wh_loss.item(), step)
        writer.add_scalar("offset_loss", off_loss.item(), step)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], step)

        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        tbar.set_postfix(total_loss="{:.4f}".format(loss.item()),
                         c_loss="{:.4f}".format(c_loss.item()),
                         wh_loss="{:.4f}".format(wh_loss.item()),
                         offset_loss="{:.4f}".format(off_loss.item()))

        # clear batch variables from memory
        del images, hms_true, whs_true, offsets_true, offset_masks_true

    return np.mean(total_loss)


def eval_one_epochs(model, val_loader, epoch, dev, writer):

    model.eval()

    total_loss = []
    total_c_loss = []
    total_wh_loss = []
    total_offset_loss = []
    write_image = True

    with torch.no_grad():
        for images, hms_true, whs_true, offsets_true, offset_masks_true in val_loader:

            # Set variables for training
            images = images.float().to(dev)
            hms_true = hms_true.float().to(dev)
            whs_true = whs_true.float().to(dev)
            offsets_true = offsets_true.float().to(dev)
            offset_masks_true = offset_masks_true.float().to(dev)

            # Get model predictions, calculate loss
            training_output = model(images, mode='train', ground_truth_data=(hms_true,
                                                                             whs_true,
                                                                             offsets_true,
                                                                             offset_masks_true))
            hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true = training_output

            loss = loss.mean()
            c_loss = c_loss.mean()
            wh_loss = wh_loss.mean()
            off_loss = off_loss.mean()

            total_loss.append(loss.item())
            total_c_loss.append(c_loss.item())
            total_wh_loss.append(wh_loss.item())
            total_offset_loss.append(off_loss.item())

            if write_image:
                write_image = False
                summary_images = get_summary_image(images,
                                                   hms_true, whs_true, offsets_true,
                                                   hms_pred, whs_pred, offsets_pred, dev)
                for i, summary_image in enumerate(summary_images):
                    writer.add_image('val_images_{}'.format(i), summary_image, global_step=epoch, dataformats="HWC")

            # clear batch variables from memory
            del images, hms_true, whs_true, offsets_true, offset_masks_true

        writer.add_scalar("val_loss", np.mean(total_loss), epoch)
        writer.add_scalar("val_c_loss", np.mean(total_c_loss), epoch)
        writer.add_scalar("val_wh_loss", np.mean(total_wh_loss), epoch)
        writer.add_scalar("val_offset_loss", np.mean(total_offset_loss), epoch)

    return np.mean(total_loss)


if __name__ == '__main__':
    remove_dir_and_create_dir(os.path.join(args.logs_dir, "weights"), is_remove=True)
    remove_dir_and_create_dir(os.path.join(args.logs_dir, "summary"), is_remove=True)

    model = get_model(args, dev)
    train_dataset, val_dataset = get_dataset(args, class_names)

    writer = SummaryWriter(os.path.join(args.logs_dir, "summary"))
    step = 0

    freeze_step = len(train_dataset) // args.freeze_batch_size
    unfreeze_step = len(train_dataset) // args.unfreeze_batch_size
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, args.learn_rate_init)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=args.freeze_epochs * freeze_step + args.unfreeze_epochs * unfreeze_step,
                                              max_lr=args.learn_rate_init,
                                              min_lr=args.learn_rate_end,
                                              warmup_steps=args.warmup_epochs * freeze_step)

    # freeze
    if args.freeze_epochs > 0:
        print("Freeze backbone and decoder, train {} epochs.".format(args.freeze_epochs))
        model.module.freeze_backbone()

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.freeze_batch_size,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.freeze_batch_size,
                                num_workers=args.num_workers, pin_memory=True)

        for epoch in range(args.freeze_epochs):
            train_loss = train_one_epochs(model, train_loader, epoch, optimizer, scheduler, dev, writer)
            val_loss = eval_one_epochs(model, val_loader, epoch, dev, writer)
            print("=> loss: {:.4f}   val_loss: {:.4f}".format(train_loss, val_loss))
            torch.save(model,
                       '{}/weights/epoch={}_loss={:.4f}_val_loss={:.4f}.pt'.
                       format(args.logs_dir, epoch, train_loss, val_loss))

    # unfreeze
    if args.unfreeze_epochs > 0:
        print("Unfreeze backbone and decoder, train {} epochs.".format(args.unfreeze_epochs))
        model.module.unfreeze_backbone()

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.unfreeze_batch_size,
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.unfreeze_batch_size,
                                num_workers=args.num_workers, pin_memory=True)

        for epoch in range(args.unfreeze_epochs):
            epoch = args.freeze_epochs + epoch
            train_loss = train_one_epochs(model, train_loader, epoch, optimizer, scheduler, dev, writer)
            val_loss = eval_one_epochs(model, val_loader, epoch, dev, writer)
            print("=> loss: {:.4f}   val_loss: {:.4f}".format(train_loss, val_loss))
            torch.save(model,
                       '{}/weights/epoch={}_loss={:.4f}_val_loss={:.4f}.pt'.
                       format(args.logs_dir, epoch, train_loss, val_loss))