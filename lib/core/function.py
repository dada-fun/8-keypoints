# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images


logger = logging.getLogger(__name__)





def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)



def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
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
            score = meta['score'].numpy()                 #评估的时候score值默认为1   无检测框的置信度

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def validate_other_model(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    #model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        # compute output
        input = input.numpy()
        #input = np.transpose(input, (0, 2, 3, 1))
        outputs = model.Run(input)
        #print('outputs::',outputs)
        outputs = np.transpose(outputs['2947/DataCrop'],(0,3,1,2))
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input_flipped = np.flip(input, 3).copy()
            #input_flipped = torch.from_numpy(input_flipped).cuda()
            #input_flipped = input_flipped.numpy()
            outputs_flipped = model.Run(input_flipped)
            outputs_flipped = np.transpose(outputs_flipped['2947/DataCrop'],(0,3,1,2))

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped,
                                       val_dataset.flip_pairs)
            #output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5
            print('output = (output + output_flipped) * 0.5')


        output = torch.from_numpy(output.copy()).cuda()

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = criterion(output, target, target_weight)

        input = torch.from_numpy(input.copy()).cuda()
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
        score = meta['score'].numpy()                 #评估的时候score值默认为1   无检测框的置信度

        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        idx += num_images

        if i % config.PRINT_FREQ == 0:
            msg = 'Test: [{0}/{1}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses, acc=acc)
            logger.info(msg)

            prefix = '{}_{}'.format(
                os.path.join(output_dir, 'val'), i
            )
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

    name_values, perf_indicator = val_dataset.evaluate(
        config, all_preds, output_dir, all_boxes, image_path,
        filenames, imgnums
    )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar(
            'valid_loss',
            losses.avg,
            global_steps
        )
        writer.add_scalar(
            'valid_acc',
            acc.avg,
            global_steps
        )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

def validate_other_model__(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    # losses = AverageMeter()
    acc = AverageMeter()

    def get_final_preds(config, batch_heatmaps, center, scale):
        import math
        from utils.transforms import transform_preds


        heatmap_height = 64
        heatmap_width = 64
        coords = []
        conf = []
        for i in range(batch_heatmaps.shape[0]):
            xy = batch_heatmaps[i, :]
            # print(xy)

            raw_x = xy[1] * heatmap_width
            raw_y = xy[0] * heatmap_height  # 找到最大位置所在行-->   x坐标值
            coords.append([raw_x, raw_y])
            conf = xy[2]

        coords = np.array(coords)
        coords = np.expand_dims(coords, axis=0)
        conf = np.array(conf)
        conf = np.expand_dims(conf, axis=0)
        print('coords',conf)
        preds = np.array(coords).copy()

        #print('center.shape:::', center.shape)
        # Transform back
        for i in range(preds.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )
        maxvals = conf

        return preds, maxvals



    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        # compute output
        # outputs = model(input)
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        # print('input.size:::',input.shape)
        # print(output_details)
        input = input.permute((0,2,3,1)).numpy()
        input = input.astype(dtype=np.float32)
        
        model.allocate_tensors()
        model.set_tensor(input_details[0]['index'], input)

        model.invoke()
        outputs = [model.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        output = np.squeeze(output)

        if config.TEST.FLIP_TEST:
            # this part is ugly, because pytorch has not supported negative index
            # input_flipped = model(input[:, :, :, ::-1])
            input = torch.from_numpy(input)
            input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            #input_flipped = torch.from_numpy(input_flipped).cuda()
            model.set_tensor(input_details[0]['index'], input_flipped)
            model.invoke()

            outputs_flipped = [model.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped,
                                       val_dataset.flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


            # feature is not aligned, shift flipped heatmap for higher accuracy
            if config.TEST.SHIFT_HEATMAP:
                output_flipped[:, :, :, 1:] = \
                    output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5

        # target = target.cuda(non_blocking=True)
        # target_weight = target_weight.cuda(non_blocking=True)

        # loss = criterion(output, target, target_weight)

        # print('input.size:::', input.shape)
        # num_images = input.size(0)
        num_images = 1
        # measure accuracy and record loss
        # losses.update(loss.item(), num_images)
        # _, avg_acc, cnt, pred = accuracy(output,
        #                                  target)

        # acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()




        preds, maxvals = get_final_preds(
            config, output, c, s)

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        # print('all_preds:::',all_preds[idx:idx + num_images,:,:])
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
        all_boxes[idx:idx + num_images, 5] = score
        image_path.extend(meta['image'])

        idx += num_images

        # if i % config.PRINT_FREQ == 0:
        #     msg = 'Test: [{0}/{1}]\t' \
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
        #           'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
        #               i, len(val_loader), batch_time=batch_time,
        #               loss=0.001, acc=acc)
        #     logger.info(msg)

            # prefix = '{}_{}'.format(
            #     os.path.join(output_dir, 'val'), i
            # )
            # save_debug_images(config, input, meta, target, pred*4, output,
            #                   prefix)

        # print('image_path:::',image_path)
        # print('all_preds:::', all_preds.shape)
    name_values, perf_indicator = val_dataset.evaluate(
        config, all_preds, output_dir, all_boxes, image_path,
        filenames, imgnums
    )

    model_name = config.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        # writer.add_scalar(
        #     'valid_loss',
        #     # losses.avg,
        #     global_steps
        # )
        # writer.add_scalar(
        #     'valid_acc',
        #     acc.avg,
        #     global_steps
        # )
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars(
                    'valid',
                    dict(name_value),
                    global_steps
                )
        else:
            writer.add_scalars(
                'valid',
                dict(name_values),
                global_steps
            )
        writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator



# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
