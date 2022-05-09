#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :  2021/07/22
# @Author  :  WB.Li
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
from torchvision import transforms

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    points = points.tolist()
    dup = img.copy()
    h, w, _ = dup.shape
    thickness = int(5 / 640 * h)

    colors = [0, 255, 0]
    for ind, (start_ind, end_ind) in enumerate(conn):
        # xmin, ymin = points[start_ind][:2]
        # xmax, ymax = points[end_ind][:2]
        ymin, xmin = points[start_ind][:2]
        ymax, xmax = points[end_ind][:2]
        cv2.line(dup,
                 (int(xmin), int(ymin)),
                 (int(xmax), int(ymax)),
                 colors,
                 thickness)

    return dup


def show_kps(image, points):
    for i, p in enumerate(points):
        x, y = p
        # color = (0, 0, 255) if (visibility>0.2) else (255, 0, 0)
        cv2.circle(image, center=(int(x), int(y)), color=(255, 0, 0), radius=3, thickness=2)
        image = cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (0, 255, 0), 1, cv2.LINE_AA)

    return image

def heatmap2point(out, heatmaps, boxes, resolution):
    pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
    # For each human, for each joint: y, x, confidence
    for i, human in enumerate(out):
        heatmaps[i] = human
        for j, joint in enumerate(human):
            pt = np.unravel_index(np.argmax(joint), (resolution[0] // 4, resolution[1] // 4))
            pts[i, j, 0] = pt[0] * 1. / (resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
            pts[i, j, 1] = pt[1] * 1. / (resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
            pts[i, j, 2] = joint[pt]

    return pts[0]

def vis_pose(img, points):
    for i, point in enumerate(points):
        x, y,sc = point
        x, y = int(x), int(y)
        if sc > 0.:
            cv2.circle(img,(y,x),2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(img,'{}'.format(i), (y,x),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)

    return img


def infer(model, img_dir):
    names = os.listdir(img_dir)
    names.sort()

    model.cpu().eval()
    resolution = (384, 288)
    nof_joints = 8
    #conn = [[5, 7], [5, 6], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16]]

    try:
        for name in names:
            import time
            t1 = time.time()
            file = os.path.join(img_dir, name)
            image_org = cv2.imread(file)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            img_org_h, img_org_w, _ = image_org.shape
            input_h, input_w = resolution[0], resolution[1]
            scale = min(input_w / img_org_w, input_h / img_org_h)  # 转换的最小比例
            nw = int(img_org_w * scale + 0.5)
            nh = int(img_org_h * scale + 0.5)
            image_org = cv2.resize(image_org, (nw, nh), interpolation=cv2.INTER_CUBIC)  # (width, height)
            left_p = int((input_w - nw) / 2)
            right_p = input_w - left_p - nw
            top_p = int((input_h - nh) / 2)
            bottom_p = input_h - top_p - nh
            image_org = cv2.copyMakeBorder(image_org, top_p, bottom_p, left_p, right_p,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
            images = transform(cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
            boxes = np.asarray([[0, 0, 288, 384]], dtype=np.float32)  # [x1, y1, x2, y2]
            heatmaps = np.zeros((1, nof_joints, 384 // 4, 288 // 4), dtype=np.float32)
            # images = images.numpy()


            pred = model(images)
            print("pred:", type(pred), pred.shape)
            pred = pred.detach().cpu().numpy()

            point_res = heatmap2point(pred, heatmaps,boxes, resolution)

            img_vim = vis_pose(image_org, point_res)
            #img_vim = draw_2d_line(img_vim, point_res, conn)
            result = img_vim[top_p:nh, left_p:left_p + nw]
            result_re = cv2.resize(result, (img_org_w, img_org_h))

            save_root = 'test_img/vis_imgs'
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_img_p = os.path.join(save_root, name)
            cv2.imwrite(save_img_p, img_vim)
            t2 = time.time()
            print("time is ", t2-t1)
    except:
        pass


        
        




if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parse_args()
    img_dir = args.dataDir

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
    checkpoint = torch.load(model_file, map_location='cpu')
    # model.module.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    print(model)

    infer(model, img_dir)



