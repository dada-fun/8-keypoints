#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2021/07/22
# @Author  :  Apei.zou
# @File    :  eval.py
# @Software:  Pycharm

import os, sys
import argparse
import importlib

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import time

base_path = os.getcwd()

sys.path.append(base_path)
sys.path.append(os.path.join(base_path, "lib"))
sys.path.append(os.path.join(base_path, "utils"))
sys.path.append(os.path.join(base_path, "models"))


from lib.config import cfg
from lib.config import update_config
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def parse_args():
    parser = argparse.ArgumentParser(description='Human Pose Training')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def time_synchronized():
    # pytorch-accurate time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def padding_img(img, dest_size=None, color=(255, 255, 255)):
    ori_h, ori_w, _ = img.shape

    if dest_size is None:
        if ori_h >= ori_w:
            dest_size = (ori_h, ori_h)
        else:
            dest_size = (ori_w, ori_w)

    if dest_size[0] < ori_w and dest_size[1] < ori_h:
        raise Exception("The dest size must small than origin image")

    w_offset = max(0, int((dest_size[0] - ori_w) // 2))
    h_offset = max(0, int((dest_size[1] - ori_h) // 2))

    dest_img = cv2.copyMakeBorder(img, h_offset, h_offset, w_offset, w_offset, cv2.BORDER_CONSTANT,
                                  color)
    dest_img = cv2.resize(dest_img, (int(dest_size[0]), int(dest_size[1])))
    return (dest_img, w_offset, h_offset)

def draw_2d_line(img, points, conn):
    """

    :param img:
    :param points:
    :param conn: [(0, 1), (1, 2), (3, 4),...]
    :return:
    """
    dup = img.copy()
    h, w, _ = dup.shape
    thickness = int(5 / 640 * h)

    colors = [0, 255, 0]
    for ind, (start_ind, end_ind) in enumerate(conn):
        xmin, ymin = points[start_ind]
        xmax, ymax = points[end_ind]
        cv2.line(dup,
                 (int(xmin), int(ymin)),
                 (int(xmax), int(ymax)),
                 colors,
                 thickness)

    return dup


if __name__ == '__main__':
    args = parse_args()

    update_config(cfg, args)
    if torch.cuda.is_available():
        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    module_name = "lib.models." + cfg.MODEL.NAME
    mod = importlib.import_module(module_name)
    model = mod.get_pose_net(cfg, is_train=False)
    model_file = args.modelDir
    checkpoint = torch.load(model_file)
    # model.module.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    cnt = 0
    avg_imread = 0
    avg_resize = 0
    avg_infer = 0
    avg_pose = 0

    heap_size = 64
    scale_size = 256
    while(1):
        cnt = cnt + 1
        img_path = './imgs/left.jpg'
        t1 = time_synchronized()
        img = cv2.imread(img_path)
        t2 = time_synchronized()
        avg_imread += t2 - t1
        print('average time imread:::', (avg_imread / cnt))

        t1 = time_synchronized()
        inp_frame, w_offset, h_offset = padding_img(img)
        tmp_img = cv2.resize(inp_frame, (scale_size, scale_size))
        pad_h, pad_w, _ = inp_frame.shape
        h_scale, w_scale = pad_h / heap_size, pad_w / heap_size
        # 归一化
        tmp_img = tmp_img / 127.5 - 1
        inp_img = tmp_img.transpose((2, 0, 1)) # h, w, c -> c, h, w
        inp_img = inp_img[np.newaxis, :, :, :] # 拓展成4维
        inp_img = inp_img.astype(np.float32)
        inp_img = torch.from_numpy(inp_img)
        t2 = time_synchronized()
        avg_resize += t2 - t1
        print('average time padding&resize:::', (avg_resize / cnt))

        t1 = time_synchronized()
        pred = model(inp_img)
        # print("pred:", type(pred), pred.shape)
        t2 = time_synchronized()
        avg_infer += t2 - t1
        print('average time infer:::', (avg_infer / cnt))

        t1 = time_synchronized()
        coords = []
        if torch.cuda.is_available():
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach().numpy()
        pred = np.squeeze(pred)
        conf = []
        for i in range(pred.shape[0]):
            heat = pred[i, :, :]
            ids = np.argmax(heat)
            raw_y = ids // heap_size
            raw_x = ids % heap_size
            y = int(raw_y * h_scale - h_offset)
            x = int(raw_x * w_scale - w_offset)
            coords.append([x, y])
            conf.append(pred[i, raw_y, raw_x])
        t2 = time_synchronized()
        avg_pose += t2 - t1
        print('average time post:::', (avg_pose / cnt))

        # 0: 'R_Ankle',1: 'R_Knee', 2: 'R_Hip',3: 'L_Hip', 4: 'L_Knee', 5: 'L_Ankle',6: 'Neck',7: 'B_Head',8: 'R_Wrist'
        # 9: 'R_Elbow', 10: 'R_Shoulder', 11: 'L_Shoulder', 12: 'L_Elbow', 13: 'L_Wrist'
        img = draw_2d_line(img, coords,[[7, 6], [6, 11], [11, 12], [12, 13], [6, 10], [10, 9], [9, 8], [3, 4], [4, 5], [2, 1],
                            [1, 0]])
        # plt.subplot(1, 2, 1)
        # plt.imshow(img[:, :, ::-1])
        img_dir = os.path.dirname(img_path)
        out_name = os.path.basename(img_path).split('.')[0] + '_out.jpg'
        out_file = os.path.join(img_dir, out_name)
        # print(out_file)
        plt.imsave(out_file, img[:, :, ::-1])

