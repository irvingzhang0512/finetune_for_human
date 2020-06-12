from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label


def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class handpose_model(nn.Module):
    def __init__(self):
        super(handpose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = [
            'conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3', 'Mconv7_stage4',
            'Mconv7_stage5', 'Mconv7_stage6'
        ]
        # stage 1
        block1_0 = OrderedDict([('conv1_1', [3, 64, 3, 1, 1]),
                                ('conv1_2', [64, 64, 3, 1, 1]),
                                ('pool1_stage1', [2, 2, 0]),
                                ('conv2_1', [64, 128, 3, 1, 1]),
                                ('conv2_2', [128, 128, 3, 1, 1]),
                                ('pool2_stage1', [2, 2, 0]),
                                ('conv3_1', [128, 256, 3, 1, 1]),
                                ('conv3_2', [256, 256, 3, 1, 1]),
                                ('conv3_3', [256, 256, 3, 1, 1]),
                                ('conv3_4', [256, 256, 3, 1, 1]),
                                ('pool3_stage1', [2, 2, 0]),
                                ('conv4_1', [256, 512, 3, 1, 1]),
                                ('conv4_2', [512, 512, 3, 1, 1]),
                                ('conv4_3', [512, 512, 3, 1, 1]),
                                ('conv4_4', [512, 512, 3, 1, 1]),
                                ('conv5_1', [512, 512, 3, 1, 1]),
                                ('conv5_2', [512, 512, 3, 1, 1]),
                                ('conv5_3_CPM', [512, 128, 3, 1, 1])])

        block1_1 = OrderedDict([('conv6_1_CPM', [128, 512, 1, 1, 0]),
                                ('conv6_2_CPM', [512, 22, 1, 1, 0])])

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict([
                ('Mconv1_stage%d' % i, [150, 128, 7, 1, 3]),
                ('Mconv2_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d' % i, [128, 22, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        return out_stage6


def npmax(array):
    # get max index of 2d array
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j


def transfer(model, model_weights):
    # transfer caffe model to pytorch which will match the layer name
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(
            weights_name.split('.')[1:])]
    return transfered_model_weights


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


class Hand(object):
    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, oriImg):
        scale_search = [0.5, 1.0, 1.5, 2.0]
        # scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre = 0.05
        multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 22))

        for m in range(len(multiplier)):
            scale = multiplier[m]
            imageToTest = cv2.resize(oriImg, (0, 0),
                                     fx=scale,
                                     fy=scale,
                                     interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = padRightDownCorner(
                imageToTest, stride, padValue)
            im = np.transpose(
                np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                (3, 2, 0, 1))
            im = im / 256. - 0.5
            im = np.ascontiguousarray(im)

            data = torch.from_numpy(im).float()
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                output = self.model(data).cpu().numpy()

            # extract outputs, resize, and remove padding
            # output 1 is heatmaps
            heatmap = np.transpose(np.squeeze(output), (1, 2, 0))
            heatmap = cv2.resize(heatmap, (0, 0),
                                 fx=stride,
                                 fy=stride,
                                 interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] -
                              pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]),
                                 interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap / len(multiplier)

        available_keypoints_cnt = 0
        all_peaks = []
        for part in range(21):
            map_ori = heatmap_avg[:, :, part]
            one_heatmap = gaussian_filter(map_ori, sigma=3)
            binary = np.ascontiguousarray(one_heatmap > thre, dtype=np.uint8)
            # 全部小于阈值
            if np.sum(binary) == 0:
                all_peaks.append([0, 0])
                continue
            available_keypoints_cnt += 1
            label_img, label_numbers = label(binary,
                                             return_num=True,
                                             connectivity=binary.ndim)
            max_index = np.argmax([
                np.sum(map_ori[label_img == i])
                for i in range(1, label_numbers + 1)
            ]) + 1
            label_img[label_img != max_index] = 0
            map_ori[label_img == 0] = 0

            y, x = npmax(map_ori)
            all_peaks.append([x, y])
        return available_keypoints_cnt, np.array(all_peaks)


class HandKeyPointCounter:
    def __init__(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = "/hdd02/zhangyiyang/pytorch-openpose/model/hand_pose_model.pth"
        self.hand_estimation = Hand(model_path=ckpt_path)

    def __call__(self, img_path):
        oriImg = cv2.imread(img_path)
        cnt, _ = self.hand_estimation(oriImg)
        return cnt


if __name__ == '__main__':
    counter = HandKeyPointCounter()
    print(counter("../../data/images/drinking/img_00003.jpg"))
