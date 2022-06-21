# -*- coding: utf-8 -*-
# @File : centernet.py
# @Author: Runist
# @Time : 2022/3/28 16:46
# @Software: PyCharm
# @Brief: CenterNet implement
from torch import nn
import math
from net.backbone.resnet import resnet50, resnet101
from core.loss import focal_loss, l1_loss


class CenterNet(nn.Module):
    def __init__(self, backbone, num_classes=20):
        """

        Args:
            backbone: string
            num_classes: int
        """
        super(CenterNet, self).__init__()

        # h, w, 3 -> h/32, w/32, 2048
        if backbone == "resnet50":
            self.backbone = resnet50(num_classes, include_top=False)
        elif backbone == "resnet101":
            self.backbone = resnet101(num_classes, include_top=False)
        else:
            raise Exception("There is no {}.".format(backbone))

        # h/32, w/32, 2048 -> h/4, w/4, 64
        self.decoder = CenterNetDecoder(2048)

        # feature height and width: h/4, w/4
        # hm channel: num_classes
        # wh channel: 2
        # offset channel: 2
        self.head = CenterNetHead(channel=64, num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head.cls_head[-2].weight.data.fill_(0)
        self.head.cls_head[-2].bias.data.fill_(-2.19)

    def freeze_backbone(self):
        """
        Freeze centernet backbone parameters.
        Returns: None

        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze centernet backbone parameters.
        Returns: None

        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def forward(self, x, **kwargs):
        x = self.backbone(x)
        x = self.decoder(x)
        hms_pred, whs_pred, offsets_pred = self.head(x)

        if kwargs.get('mode') == "train":
            # Change model outputs tensor order
            hms_pred = hms_pred.permute(0, 2, 3, 1)
            whs_pred = whs_pred.permute(0, 2, 3, 1)
            offsets_pred = offsets_pred.permute(0, 2, 3, 1)

            hms_true, whs_true, offsets_true, offset_masks_true = kwargs.get('ground_truth_data')

            c_loss = focal_loss(hms_pred, hms_true)
            wh_loss = 0.1 * l1_loss(whs_pred, whs_true, offset_masks_true)
            off_loss = l1_loss(offsets_pred, offsets_true, offset_masks_true)

            loss = c_loss + wh_loss + off_loss

            # Using 3x3 kernel max pooling to filter the maximum response of heatmap
            hms_true = hms_true.permute(0, 3, 1, 2)
            hms_pred = hms_pred.permute(0, 3, 1, 2)
            hms_true = CenterNetPoolingNMS(kernel=3)(hms_true)
            hms_pred = CenterNetPoolingNMS(kernel=3)(hms_pred)
            hms_true = hms_true.permute(0, 2, 3, 1)
            hms_pred = hms_pred.permute(0, 2, 3, 1)

            return hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true
        else:
            hms_pred = CenterNetPoolingNMS(kernel=3)(hms_pred)
            return hms_pred, whs_pred, offsets_pred


class CenterNetDecoder(nn.Module):
    def __init__(self, in_channels, bn_momentum=0.1):
        super(CenterNetDecoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.in_channels = in_channels
        self.deconv_with_bias = False

        # h/32, w/32, 2048 -> h/16, w/16, 256 -> h/8, w/8, 128 -> h/4, w/4, 64
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            num_filter = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=num_filter,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_filter
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class CenterNetHead(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(CenterNetHead, self).__init__()

        # heatmap
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # bounding boxes height and width
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)

        return hm, wh, offset


class CenterNetPoolingNMS(nn.Module):
    def __init__(self, kernel=3):
        """
        To replace traditional nms method. Input is heatmap, the num of channel is num_classes,
        So one object center has strongest response, where use torch.max(heatmap, dim=-1), it only
        filter single pixel max value, the neighbour pixel still have strong response, so we should
        use max pooling stride=1 to filter this fake center point.
        Args:
            kernel: max pooling kernel size
        """
        super(CenterNetPoolingNMS, self).__init__()
        self.pad = (kernel - 1) // 2
        self.max_pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=(kernel - 1) // 2)

    def forward(self, x):
        xmax = self.max_pool(x)
        keep = (xmax == x).float()

        return x * keep
