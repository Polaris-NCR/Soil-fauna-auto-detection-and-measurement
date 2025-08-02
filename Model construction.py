import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from osgeo import gdal
from skimage.draw import disk
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def vgg_block(num_convs, in_channels, out_channels):
    """
    vgg block: Conv2d ReLU MaxPool2d
    """
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))
    return blk


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        features = []
        features.extend(vgg_block(2, 3, 64))
        features.extend(vgg_block(2, 64, 128))
        features.extend(vgg_block(3, 128, 256))
        self.index_pool3 = len(features)
        features.extend(vgg_block(3, 256, 512))
        self.index_pool4 = len(features)
        features.extend(vgg_block(3, 512, 512))

        self.features = nn.Sequential(*features)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)

        if pretrained:
            pretrained_model = vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.features.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.features.load_state_dict(new_dict)

    def forward(self, x):
        pool3 = self.features[:self.index_pool3](x)
        pool4 = self.features[self.index_pool3:self.index_pool4](pool3)

        pool5 = self.features[self.index_pool4:](pool4)

        conv6 = self.relu(self.conv6(pool5))
        conv7 = self.relu(self.conv7(conv6))

        return pool3, pool4, conv7


class FCN(nn.Module):
    def __init__(self, num_classes, backbone='vgg'):
        super(FCN, self).__init__()
        if backbone == 'vgg':
            self.features = VGG16()

        self.scores1 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.scores2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.scores3 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        pool3, pool4, conv7 = self.features(x)

        conv7 = self.relu(self.scores1(conv7))

        pool4 = self.relu(self.scores2(pool4))

        pool3 = self.relu(self.scores3(pool3))

        conv7_2x = F.interpolate(self.upsample_2x(conv7),
                                 size=(pool4.size(2), pool4.size(3)))
        s = conv7_2x + pool4

        s = F.interpolate(self.upsample_2x(s), size=(pool3.size(2), pool3.size(3)))
        s = pool3 + s

        out_8s = F.interpolate(self.upsample_8x(s), size=(h, w))

        return self.sigmoid(out_8s)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        h, w = x.size()[2:]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x5 = self.gap(x)
        x5 = F.interpolate(x5, size=(h, w), mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.conv(x)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(DeepLabV3, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            self._make_layer(64, 128, 2, stride=1),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=1, dilation=2)
        )

        self.aspp = ASPP(in_channels=512, out_channels=256)

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

        self.up = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dilation=1):
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          stride=stride, padding=dilation,
                          dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        )
        for _ in range(1, blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3,
                              padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)

        x = self.aspp(x)

        x = self.classifier(x)

        x = self.up(x)

        return self.sigmoid(x)


class SegNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(SegNet, self).__init__()

        self.encoder_conv = nn.ModuleList([

            nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])

        self.decoder_conv = nn.ModuleList([

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            )
        ])

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        encoder_outs = []
        pool_indicies = []

        for i, block in enumerate(self.encoder_conv):
            x = block(x)
            encoder_outs.append(x)
            x, indices = self.pool(x)
            pool_indicies.append(indices)

        for i, block in enumerate(self.decoder_conv):
            x = self.unpool(x, pool_indicies[-i - 1], output_size=encoder_outs[-i - 1].size())
            x = block(x)

        # 最终输出
        return self.sigmoid(x)


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        b, c, _, _ = x.size()

        # 平均池化路径
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        # 最大池化路径
        max_out = self.fc(self.max_pool(x).view(b, c))

        # 自适应融合双路径信息
        out = self.alpha * avg_out + (1 - self.alpha) * max_out
        return out.view(b, c, 1, 1)


class AttnDoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        return x * self.attn(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        features = init_features

        self.encoder1 = AttnDoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = AttnDoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = AttnDoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = AttnDoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = AttnDoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.decoder4 = AttnDoubleConv(features * 16, features * 8)  # 拼接后通道数翻倍
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.decoder3 = AttnDoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.decoder2 = AttnDoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.decoder1 = AttnDoubleConv(features * 2, features)

        self.conv_out = nn.Conv2d(features, out_channels, 1)

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(out_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        # print(enc1.shape)
        enc2 = self.encoder2(self.pool1(enc1))
        # print(enc2.shape)
        enc3 = self.encoder3(self.pool2(enc2))
        # print(enc3.shape)
        enc4 = self.encoder4(self.pool3(enc3))
        # print(enc4.shape)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out = self.conv_out(dec1)

        return self.sigmoid(out)


def cal_metrics(preds, targets):
    targets = targets.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()

    accuracy, precision, recall, F1, IOU_true, IOU_false, mIOU, mAP = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(targets.shape[0]):
        metrix = confusion_matrix(targets[i].flatten(), preds[i].flatten())
        if metrix.shape == (1, 1):
            metrix = np.array([[metrix[0, 0], 0], [0, metrix[0, 0]]])
        TP = metrix[1, 1]  # True Positive
        TN = metrix[0, 0]  # True Negative
        FP = metrix[0, 1]  # False Positive
        FN = metrix[1, 0]  # False Negative

        accuracy += (TP + TN) / (TP + TN + FP + FN)
        precision += TP / (TP + FP) if (TP + FP) != 0 else 0
        recall += TP / (TP + FN) if (TP + FN) != 0 else 0
        F1 += 2 * precision * recall / (precision + recall)
        IOU1 = TP / (TP + FP + FN)
        IOU0 = TN / (TN + FP + FN)
        IOU_true += IOU1
        IOU_false += IOU0
        mIOU += (IOU0 + IOU1) / 2

        y_true = targets[i].flatten()
        y_pred_prob = preds[i].flatten()

        ap_pos = average_precision_score(y_true, y_pred_prob)

        mAP += ap_pos

    accuracy /= targets.shape[0]
    precision /= targets.shape[0]
    recall /= targets.shape[0]
    F1 /= targets.shape[0]
    IOU_true /= targets.shape[0]
    IOU_false /= targets.shape[0]
    mIOU /= targets.shape[0]
    mAP /= targets.shape[0]

    return accuracy, precision, recall, F1, IOU_true, IOU_false, mIOU, mAP


def train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, lambda_acc=0.5, num_epochs=100,
          print_interval=10):
    history = {
        'train_loss_history': [], 'val_loss_history': [],
        'train_acc_history': [], 'val_acc_history': [],
        'train_precision_history': [], 'val_precision_history': [],
        'train_recall_history': [], 'val_recall_history': [],
        'train_F1_history': [], 'val_F1_history': [],
        'train_IOU_true_history': [], 'val_IOU_true_history': [],
        'train_IOU_false_history': [], 'val_IOU_false_history': [],
        'train_mIOU_history': [], 'val_mIOU_history': [],
        'train_AP_history': [], 'val_AP_history': []
    }

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_metrics = {'train_acc': 0.0, 'train_precision': 0.0, 'train_recall': 0.0, 'train_f1': 0.0,
                         'train_IOU_true': 0.0, 'train_IOU_false': 0.0, 'train_mIOU': 0.0, 'train_AP': 0.0}
        val_metrics = {'val_acc': 0.0, 'val_precision': 0.0, 'val_recall': 0.0, 'val_f1': 0.0, 'val_IOU_true': 0.0,
                       'val_IOU_false': 0.0, 'val_mIOU': 0.0, 'val_AP': 0.0}
        print('Epoch: %d' % int(epoch + 1))
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            model.train()
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions = model(data)

            preds = (predictions > 0.5).float()
            # train_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)
            # bce_loss = criterion(predictions, targets)
            # accuracy, precision, recall, f1_score = cal_metrics(preds, targets)
            accuracy, precision, recall, f1_score, IOU_true, IOU_false, mIOU, mAP = cal_metrics(preds, targets)
            train_metrics['train_acc'] += accuracy
            train_metrics['train_precision'] += precision
            train_metrics['train_recall'] += recall
            train_metrics['train_f1'] += f1_score
            train_metrics['train_IOU_true'] += IOU_true
            train_metrics['train_IOU_false'] += IOU_false
            train_metrics['train_mIOU'] += mIOU
            train_metrics['train_AP'] += mAP
            # loss = bce_loss + (1 - accuracy) * lambda_acc
            loss = criterion(predictions, targets)
            loss.backward()
            train_loss += loss.item()

            optimizer.step()

        history['train_loss_history'].append(train_loss / (batch_idx + 1))
        history['train_acc_history'].append(train_metrics['train_acc'] / (batch_idx + 1))
        history['train_precision_history'].append(train_metrics['train_precision'] / (batch_idx + 1))
        history['train_F1_history'].append(train_metrics['train_f1'] / (batch_idx + 1))
        history['train_recall_history'].append(train_metrics['train_recall'] / (batch_idx + 1))
        history['train_IOU_true_history'].append(train_metrics['train_IOU_true'] / (batch_idx + 1))
        history['train_IOU_false_history'].append(train_metrics['train_IOU_false'] / (batch_idx + 1))
        history['train_mIOU_history'].append(train_metrics['train_mIOU'] / (batch_idx + 1))
        history['train_AP_history'].append(train_metrics['train_AP'] / (batch_idx + 1))

        scheduler.step()

        with torch.no_grad():
            for val_idx, (inputs, targets) in enumerate(val_loader):
                model.eval()
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                val_loss += criterion(output, targets).item()

                preds = (output > 0.5).float()
                # accuracy, precision, recall, f1_score = cal_metrics(preds, targets)
                accuracy, precision, recall, F1, IOU_true, IOU_false, mIOU, mAP = cal_metrics(preds, targets)
                val_metrics['val_acc'] += accuracy
                val_metrics['val_precision'] += precision
                val_metrics['val_recall'] += recall
                val_metrics['val_f1'] += f1_score
                val_metrics['val_IOU_true'] += IOU_true
                val_metrics['val_IOU_false'] += IOU_false
                val_metrics['val_mIOU'] += mIOU
                val_metrics['val_AP'] += mAP

            history['val_loss_history'].append(val_loss / (val_idx + 1))
            history['val_acc_history'].append(val_metrics['val_acc'] / (val_idx + 1))
            history['val_precision_history'].append(val_metrics['val_precision'] / (val_idx + 1))
            history['val_F1_history'].append(val_metrics['val_f1'] / (val_idx + 1))
            history['val_recall_history'].append(val_metrics['val_recall'] / (val_idx + 1))
            history['val_IOU_true_history'].append(val_metrics['val_IOU_true'] / (val_idx + 1))
            history['val_IOU_false_history'].append(val_metrics['val_IOU_false'] / (val_idx + 1))
            history['val_mIOU_history'].append(val_metrics['val_mIOU'] / (val_idx + 1))
            history['val_AP_history'].append(val_metrics['val_AP'] / (val_idx + 1))

        if (epoch + 1) % print_interval == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {history['train_loss_history'][-1]:.4f} | "
                  f"Val Loss: {history['val_loss_history'][-1]:.4f} | "
                  f"Train acc: {history['train_acc_history'][-1]:.4f} | "
                  f"Val acc: {history['val_acc_history'][-1]:.4f} | "
                  f"Train IOU: {history['train_IOU_true_history'][-1]:.4f} | "
                  f"Val IOU: {history['val_mIOU_history'][-1]:.4f} | "
                  f"Train AP: {history['train_AP_history'][-1]:.4f} | "
                  f"Val AP: {history['val_AP_history'][-1]:.4f}"
                  )

    return model, history


def vis_hitory(history):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16

    # --------------------- a) Loss ---------------------
    plt.subplot(221)
    plt.plot(history['train_loss_history'], label='Train Loss')
    plt.plot(history['val_loss_history'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(prop={'family': 'Arial', 'size': 14})
    # 添加编号标签 (左上角，相对坐标)
    plt.text(0.02, 0.95, 'a)', transform=plt.gca().transAxes,
             fontsize=16, fontweight='bold', va='top')

    # --------------------- b) Accuracy ---------------------
    plt.subplot(222)
    plt.plot(history['train_acc_history'], label='Train Accuracy')
    plt.plot(history['val_acc_history'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(prop={'family': 'Arial', 'size': 14})
    plt.text(0.02, 0.95, 'b)', transform=plt.gca().transAxes,
             fontsize=16, fontweight='bold', va='top')

    # --------------------- c) F1 ---------------------
    plt.subplot(223)
    plt.plot(history['train_AP_history'], label='Train mAP')
    plt.plot(history['val_AP_history'], label='Val mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend(prop={'family': 'Arial', 'size': 14})
    plt.text(0.02, 0.95, 'c)', transform=plt.gca().transAxes,
             fontsize=16, fontweight='bold', va='top')

    # --------------------- d) IOU ---------------------
    plt.subplot(224)
    plt.plot(history['train_mIOU_history'], label='Train IOU')
    plt.plot(history['val_mIOU_history'], label='Val IOU')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.legend(prop={'family': 'Arial', 'size': 14})
    plt.text(0.02, 0.95, 'd)', transform=plt.gca().transAxes,
             fontsize=16, fontweight='bold', va='top')


    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def RGB_to_gray(raster, label_filter=False, savepath=None, ga=None):

    if label_filter:
        array = raster[0]
        array[array == 255] = 0
        raster[0] = array

    raster_gray = 0.299 * raster[0] + 0.587 * raster[1] + 0.114 * raster[2]
    raster_gray = np.uint8(raster_gray)
    if savepath is not None:
        ga.SaveArray(raster_gray, savepath)
    return raster_gray


def crop_image_and_pre(img_path, module, device, crop_size=64, stride=64, output_dir=None):
    img = gdal.Open(img_path)
    raster = img.ReadAsArray()
    if raster.shape[0] == 4:
        raster = raster[:3]

    indices = np.where(raster[0] == 255)
    gray = RGB_to_gray(raster, label_filter=True, savepath=None)
    # musked_image = build_circle_musk(gray, savepath=None)

    h, w = raster.shape[1:]
    # base_name = os.path.splitext(os.path.basename(img_path))[0]

    mask_arr = np.zeros_like(raster[0])
    out_arr = np.zeros_like(raster[0])

    # 计算裁剪位置
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if (y + crop_size) > h or (x + crop_size) > w:
                continue

            img_crop = raster[:, y:y + crop_size, x:x + crop_size]

            tensor = torch.tensor(img_crop).float().unsqueeze(0).to(device)
            output = module(tensor)
            pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
            pred_mask[pred_mask == 1.0] = 255
            mask_arr[y:y + crop_size, x:x + crop_size] = pred_mask

    mask_arr[indices[0], indices[1]] = 0
    height, width = mask_arr.shape[0], mask_arr.shape[1]
    center = (height // 2, width // 2)
    index = np.argmax(mask_arr[0] == 0)
    radius = np.sqrt(np.power(center[0], 2) + np.power((center[1] - index), 2))
    radius = math.floor(radius) - 100
    rr, cc = disk(center, radius, shape=(height, width))

    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    out_arr[mask] = mask_arr[mask]
    raster = raster.transpose(1, 2, 0)
    return raster, mask_arr, out_arr

