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
from torchvision import transforms

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

def eval():
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        # compute output
        outputs = model(input)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped.cpu().numpy(),
                                       val_dataset.flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        num_images = input.size(0)
        # measure accuracy and record loss
        losses.update(loss.item(), num_images)
        _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                         target.cpu().numpy())

        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()  # 评估的时候score值默认为1   无检测框的置信度

        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        idx += num_images


def pred(model, img):

    inp_frame, w_offset, h_offset = padding_img(img)
    tmp_img = cv2.resize(inp_frame, (scale_size, scale_size))
    pad_h, pad_w, _ = inp_frame.shape
    h_scale, w_scale = pad_h / heap_size, pad_w / heap_size
    # 归一化
    tmp_img = tmp_img / 127.5 - 1
    inp_img = tmp_img.transpose((2, 0, 1))  # h, w, c -> c, h, w
    inp_img = inp_img[np.newaxis, :, :, :]  # 拓展成4维
    inp_img = inp_img.astype(np.float32)
    inp_img = torch.from_numpy(inp_img)
    t2 = time_synchronized()
    avg_resize += t2 - t1
    print('average time padding&resize:::', (avg_resize / cnt))

def show_kps(image, points):
    for i, p in enumerate(points):
        x, y = p
        # color = (0, 0, 255) if (visibility>0.2) else (255, 0, 0)
        cv2.circle(image, center=(int(x), int(y)), color=(255, 0, 0), radius=3, thickness=2)
        image = cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (0, 255, 0), 1, cv2.LINE_AA)

    return image

def infer(model, img_dir, out_path):
    names = os.listdir(img_dir)
    names.sort()
    FPS = []
    model.cpu().eval()
    scale_size = (192, 256)
    heap_size = [48, 64]
    for name in names:
        file = os.path.join(img_dir, name)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inp_frame, w_offset, h_offset = padding_img(img)
        tmp_img = cv2.resize(inp_frame, scale_size)
        pad_h, pad_w, _ = inp_frame.shape
        h_scale, w_scale = pad_h / heap_size[1], pad_w / heap_size[0]
        # 归一化
        # tmp_img = tmp_img / 127.5 - 1

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        tmp_img = img_transform(tmp_img).numpy()
        # inp_img = tmp_img.transpose((2, 0, 1))  # h, w, c -> c, h, w
        inp_img = tmp_img
        inp_img = inp_img[np.newaxis, :, :, :]  # 拓展成4维
        inp_img = inp_img.astype(np.float32)
        inp_img = torch.from_numpy(inp_img)

        # plt.imshow(tmp_img)
        # plt.show()
        pred = model(inp_img)
        # print("pred:", type(pred), pred.shape)

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
            # raw_y = ids // heap_size[1]
            # raw_x = ids % heap_size[0]
            raw_y = ids // heap_size[1]
            raw_x = ids % heap_size[0]
            y = int(raw_y * h_scale - h_offset)
            x = int(raw_x * w_scale - w_offset)
            coords.append([x, y])
            conf.append(pred[i, raw_y, raw_x])
        # conf
        img = show_kps(img, coords)
        print('conf:::', conf)

        plt.imshow(img)
        plt.show()




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    args = parse_args()
    img_dir = args.dataDir
    out_path = r'D:\project\Pycode\dataset\test20211208_data\hrnet_show_256'

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
    # model = torch.load(model_file, map_location='cpu')

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    #     model = torch.nn.DataParallel(model)


    infer(model, img_dir, out_path)



