#!/usr/bin/env python3
"""
Full modified evaluation script for YolactEdge with buffering control.
"""

from yolact_edge.data import COCODetection, YoutubeVIS, get_label_map, MEANS, COLORS
from yolact_edge.data import cfg, set_cfg, set_dataset
from yolact_edge.yolact import Yolact
from yolact_edge.utils.augmentations import BaseTransform, BaseTransformVideo, FastBaseTransform, Resize
from yolact_edge.utils.functions import MovingAverage, ProgressBar
from yolact_edge.layers.box_utils import jaccard, center_size, mask_iou
from yolact_edge.utils import timer
from yolact_edge.utils.functions import SavePath
from yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from yolact_edge.utils.tensorrt import convert_to_tensorrt

import pycocotools
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import logging
import math

from multiprocessing.pool import ThreadPool
from queue import Queue

##############################################
# Utility Functions and Argument Parsing
##############################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default=None, type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='File to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, resume mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='Maximum number of images to consider. Use -1 for all.')
    parser.add_argument('--eval_stride', default=5, type=int,
                        help='The default frame eval stride.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='Dump detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='Output file for coco bbox results.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='Output file for coco mask results.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='Dump detections for the web viewer.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='Path to dump web detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='Visualize lincomb masks if the config uses them.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Run in display mode without showing images.')
    parser.add_argument('--fast_eval', default=False, dest='fast_eval', action='store_true',
                        help='Skip warping frames when there is no GT annotations.')
    parser.add_argument('--deterministic', default=False, dest='deterministic', action='store_true',
                        help='Enable deterministic flags in PyTorch.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='Seed for shuffling. Only affects non-cuda parts.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs debugging info for mask protos.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks.')
    parser.add_argument('--image', default=None, type=str,
                        help='Path to an image for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='Input->output folder for images.')
    parser.add_argument('--video', default=None, type=str,
                        help='Path to a video or a digit for webcam index.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='Number of frames to evaluate in parallel.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Threshold under which detections will be ignored.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='Override the dataset specified in the config.')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Only do object detection without the mask branch.')
    parser.add_argument('--yolact_transfer', dest='yolact_transfer', action='store_true',
                        help='For splitting pretrained FPN weights for YOLACT models.')
    parser.add_argument('--coco_transfer', dest='coco_transfer', action='store_true',
                        help='[Deprecated] For splitting pretrained FPN weights.')
    parser.add_argument('--drop_weights', default=None, type=str,
                        help='Comma-separated list of weights to drop from the model.')
    parser.add_argument('--calib_images', default=None, type=str,
                        help='Directory of images for TensorRT INT8 calibration.')
    parser.add_argument('--trt_batch_size', default=1, type=int,
                        help='Max batch size to use during TRT conversion; must be >= inference batch size.')
    parser.add_argument('--disable_tensorrt', default=False, dest='disable_tensorrt', action='store_true',
                        help='Disable TensorRT optimization.')
    parser.add_argument('--use_fp16_tensorrt', default=False, dest='use_fp16_tensorrt', action='store_true',
                        help='Use FP16 optimization instead of INT8 for TensorRT.')
    parser.add_argument('--use_tensorrt_safe_mode', default=False, dest='use_tensorrt_safe_mode', action='store_true',
                        help='Enable safe mode for TensorRT engine issues.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False, benchmark=False, no_sort=False, mask_proto_debug=False, crop=True, detect=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Inverted category lookup
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_coco_cats():
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id

def get_coco_cat(transformed_cat_id):
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    return coco_cats_inv[coco_cat_id]

##############################################
# Display and Evaluation Functions
##############################################

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop, score_threshold=args.score_threshold)
        torch.cuda.synchronize()
    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            masks = t[3][:args.top_k]
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break
    if num_dets_to_consider == 0:
        return (img_gpu * 255).byte().cpu().numpy()
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color
    if args.display_masks and cfg.eval_mask_branch:
        masks = masks[:num_dets_to_consider, :, :, None]
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1,1,1,3)
                            for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1,1,1,3) * colors * mask_alpha
        inv_alph_masks = masks * (-mask_alpha) + 1
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)
        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]
            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1
                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]
                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return img_numpy

def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k].cpu().numpy() for x in t]
    with timer.env('Sync'):
        torch.cuda.synchronize()

##############################################
# Detections and mAP Calculation
##############################################

class Detections:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []
    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        bbox = [round(float(x)*10)/10 for x in bbox]
        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })
    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]
        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    def dump_web(self):
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                       'use_yolo_regressors', 'use_prediction_matching',
                       'train_masks']
        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }
        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}
        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })
        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)


def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h*w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes  , gt_boxes   = split(gt_boxes)
                crowd_masks  , gt_masks   = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores
        masks = masks.view(-1, h*w).cuda()
        boxes = boxes.cuda()


    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   box_scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], mask_scores[i])
            return
    
    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt   = len(gt_classes)

        mask_iou_cache = _mask_iou(masks, gt_masks)
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        box_indices = sorted(range(num_pred), key=lambda i: -box_scores[i])
        mask_indices = sorted(box_indices, key=lambda i: -mask_scores[i])

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(),
                     lambda i,j: crowd_bbox_iou_cache[i,j].item(),
                     lambda i: box_scores[i], box_indices),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(),
                     lambda i,j: crowd_mask_iou_cache[i,j].item(),
                     lambda i: mask_scores[i], mask_indices)
        ]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])
        
        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func, score_func, indices in iou_types:
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in indices:
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(score_func(i), True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(score_func(i), False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)

##############################################
# Evaluation Functions (Image, Video, etc.)
##############################################

def evalimage(net:Yolact, path:str, save_path:str=None, detections:Detections=None, image_id=None):
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    if cfg.flow.warp_mode != 'none':
        assert False, "Evaluating the image with a video-based model."
    extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}
    preds = net(batch, extras=extras)["pred_outs"]
    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    if args.output_coco_json:
        with timer.env('Postprocess'):
            _, _, h, w = batch.size()
            classes, scores, boxes, masks = postprocess(preds, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    detections.add_bbox(image_id, classes[i], boxes[i,:], scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], scores[i])
    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]
    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)

def evalimages(net:Yolact, input_folder:str, output_folder:str, detections:Detections=None):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    print()
    for i, p in enumerate(Path(input_folder).glob('*')):
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)
        evalimage(net, path, out_path, detections=detections, image_id=str(i))
        print(path + ' -> ' + out_path)
    print('Done.')

class CustomDataParallel(torch.nn.DataParallel):
    def gather(self, outputs, output_device):
        return sum(outputs, [])

##############################################
# Modified Video Evaluation with Buffer Control
##############################################

def evalvideo(net:Yolact, path:str):
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    vid = cv2.VideoCapture(int(path)) if is_webcam else cv2.VideoCapture(path)
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)
    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(400)
    fps = 0
    frame_time_target = 1 / vid.get(cv2.CAP_PROP_FPS)
    running = True
    frame_idx = 0
    every_k_frames = 5
    moving_statistics = {"conf_hist": []}

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        return [vid.read()[1] for _ in range(args.video_multiframe)]

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        nonlocal frame_idx
        with torch.no_grad():
            frames, imgs = inp
            if frame_idx % every_k_frames == 0 or cfg.flow.warp_mode == 'none':
                extras = {"backbone": "full", "interrupt": False, "keep_statistics": True,
                          "moving_statistics": moving_statistics}
                net_outs = net(imgs, extras=extras)
                moving_statistics["feats"] = net_outs["feats"]
                moving_statistics["lateral"] = net_outs["lateral"]
            else:
                extras = {"backbone": "partial", "interrupt": False, "keep_statistics": False,
                          "moving_statistics": moving_statistics}
                net_outs = net(imgs, extras=extras)
            frame_idx += 1
            return frames, net_outs["pred_outs"]

    def prep_frame(inp):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True)

    frame_buffer = Queue()
    video_fps = 0

    # The function to play video frames
    def play_video():
        nonlocal frame_buffer, running, video_fps, is_webcam
        video_frame_times = MovingAverage(100)
        frame_time_stabilizer = frame_time_target
        last_time = None
        stabilizer_step = 0.0005
        while running:
            frame_time_start = time.time()
            if not frame_buffer.empty():
                next_time = time.time()
                if last_time is not None:
                    video_frame_times.add(next_time - last_time)
                    video_fps = 1 / video_frame_times.get_avg()
                cv2.imshow(path, frame_buffer.get())
                last_time = next_time
            if cv2.waitKey(1) == 27:  # Press Escape to exit
                running = False
            buffer_size = frame_buffer.qsize()
            if buffer_size < args.video_multiframe:
                frame_time_stabilizer += stabilizer_step
            elif buffer_size > args.video_multiframe:
                frame_time_stabilizer -= stabilizer_step
                if frame_time_stabilizer < 0:
                    frame_time_stabilizer = 0
            new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
            next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
            target_time = frame_time_start + next_frame_target - 0.001
            while time.time() < target_time:
                time.sleep(0.001)

    extract_frame = lambda x, i: (x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

    print('Initializing model... ', end='')
    eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')
    
    # Sequence of functions to process a frame
    sequence = [prep_frame, eval_network, transform_frame]
    n_threads = 4  # You can adjust this as needed
    pool = ThreadPool(processes=n_threads)
    print("Number of threads: {}".format(n_threads))
    pool.apply_async(play_video)

    active_frames = []
    inference_times = []
    MAX_BUFFER_SIZE = 10  # Maximum allowed frames in the buffer

    print()
    while vid.isOpened() and running:
        start_time = time.time()
        # NEW CODE: Wait if frame_buffer size is too high
        while frame_buffer.qsize() > MAX_BUFFER_SIZE:
            time.sleep(0.01)

        next_frames = pool.apply_async(get_next_frame, args=(vid,))
        for frame in active_frames:
            frame['value'] = pool.apply_async(sequence[frame['idx']], args=(frame['value'],))
        for frame in active_frames:
            if frame['idx'] == 0:
                frame_buffer.put(frame['value'].get())
        active_frames = [x for x in active_frames if x['idx'] > 0]
        for frame in list(reversed(active_frames)):
            frame['value'] = frame['value'].get()
            frame['idx'] -= 1
            if frame['idx'] == 0:
                active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, args.video_multiframe)]
                frame['value'] = extract_frame(frame['value'], 0)
        active_frames.append({'value': next_frames.get(), 'idx': len(sequence) - 1})
        inference_time = time.time() - start_time
        frame_times.add(inference_time)
        inference_times.append(inference_time)
        fps = args.video_multiframe / frame_times.get_avg()
        np.save(args.video, np.asarray(inference_times))
        print('\rProcessing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d    ' %
              (fps, video_fps, frame_buffer.qsize()), end='')
    cleanup_and_exit()

def savevideo(net:Yolact, in_path:str, out_path:str):
    vid = cv2.VideoCapture(in_path)
    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
    transform = FastBaseTransform()
    frame_times = MovingAverage()
    progress_bar = ProgressBar(30, num_frames)
    frame_idx = 0
    every_k_frames = 5
    moving_statistics = {"conf_hist": []}
    try:
        for i in range(num_frames):
            timer.reset()
            frame_idx = i
            with timer.env('Video'):
                frame = torch.from_numpy(vid.read()[1]).cuda().float()
                batch = transform(frame.unsqueeze(0))
                if frame_idx % every_k_frames == 0 or cfg.flow.warp_mode == 'none':
                    extras = {"backbone": "full", "interrupt": False, "keep_statistics": True,
                              "moving_statistics": moving_statistics}
                    with torch.no_grad():
                        net_outs = net(batch, extras=extras)
                    moving_statistics["feats"] = net_outs["feats"]
                    moving_statistics["lateral"] = net_outs["lateral"]
                else:
                    extras = {"backbone": "partial", "interrupt": False, "keep_statistics": False,
                              "moving_statistics": moving_statistics}
                    with torch.no_grad():
                        net_outs = net(batch, extras=extras)
                preds = net_outs["pred_outs"]
                processed = prep_display(preds, frame, None, None, undo_transform=False, class_color=True)
                out.write(processed)
            if i > 1:
                frame_times.add(timer.total_time())
                fps = 1 / frame_times.get_avg()
                progress = (i+1) / num_frames * 100
                progress_bar.set_val(i+1)
                print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), i+1, num_frames, progress, fps), end='')
    except KeyboardInterrupt:
        print('Stopping early.')
    vid.release()
    out.release()
    print()

def evaluate(net:Yolact, dataset, train_mode=False, train_cfg=None):
    net.detect.use_fast_nms = args.fast_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    detections = None
    if args.output_coco_json and (args.image or args.images):
        from collections import OrderedDict
        detections = Detections()
        prep_coco_cats()
    if args.image is not None:
        if ':' in args.image:
            inp, out = args.image.split(':')
            evalimage(net, inp, out, detections=detections, image_id="0")
        else:
            evalimage(net, args.image, detections=detections, image_id="0")
        if args.output_coco_json:
            detections.dump()
        return
    elif args.images is not None:
        inp, out = args.images.split(':')
        evalimages(net, inp, out, detections=detections)
        if args.output_coco_json:
            detections.dump()
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')
            savevideo(net, inp, out)
        else:
            evalvideo(net, args.video)
        return
    frame_times = MovingAverage(max_window_size=100000)
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    if dataset.name == "YouTube VIS":
        dataset_size = len(dataset)
    progress_bar = ProgressBar(30, dataset_size)
    print()
    if not args.display and not args.benchmark:
        ap_data = {
            'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }
        detections = Detections()
    else:
        timer.disable('Load Data')
        timer.disable('Copy')
    dataset_indices = list(range(len(dataset)))
    if args.shuffle:
        random.shuffle(dataset_indices)
    elif not args.no_sort:
        hashed = [badhash(x) for x in dataset.ids]
        dataset_indices.sort(key=lambda x: hashed[x])
    dataset_indices = dataset_indices[:dataset_size]
    try:
        if dataset.name == "YouTube VIS":
            timer.enable_all()
            if train_cfg is None:
                train_cfg = cfg
            from yolact_edge.data.youtube_vis import YoutubeVISEval, collate_fn_youtube_vis_eval
            eval_dataset = YoutubeVISEval(dataset, dataset_indices, args.max_images)
            data_loader = torch.utils.data.DataLoader(eval_dataset, num_workers=1, shuffle=False,
                                                      collate_fn=collate_fn_youtube_vis_eval)
            data_loader_iter = iter(data_loader)
            for it, video_idx in enumerate(dataset_indices):
                with timer.env('Load Data'):
                    video_frames_data = next(data_loader_iter)
                    if video_frames_data is None: continue
                    video_frames, extra_data = video_frames_data
                frame_eval_stride = args.eval_stride
                num_passes = 5 if not args.benchmark and train_cfg.dataset.use_all_frames else 1
                for pass_idx in range(num_passes):
                    meet_annot = False
                    out_recorder = []
                    moving_statistics = {"aligned_feats": [], "conf_hist": []}
                    for frame_seq_idx, (extra, (img, gt, gt_masks, h, w, num_crowd)) in enumerate(zip(extra_data, video_frames)):
                        timer.reset()
                        frame_idx, annot_idx = extra['idx']
                        with timer.env('Load Data'):
                            batch = Variable(img.unsqueeze(0))
                            if args.cuda:
                                batch = batch.cuda()
                        if train_cfg.flow is not None:
                            if frame_idx % frame_eval_stride == pass_idx:
                                meet_annot = True
                                with timer.env('Network Extra'):
                                    extras = {"backbone": "full", "keep_statistics": True,
                                              "moving_statistics": moving_statistics}
                                    gt_forward_out = net(batch, extras=extras)
                                    moving_statistics["feats"] = gt_forward_out["feats"]
                                    moving_statistics["lateral"] = gt_forward_out["lateral"]
                                    moving_statistics["images"] = batch
                            if annot_idx == -1 and args.fast_eval:
                                continue
                            if frame_idx % frame_eval_stride != pass_idx and meet_annot:
                                with timer.env('Network Extra'):
                                    extras = {"backbone": "partial", "moving_statistics": moving_statistics}
                                    forward_out = net(batch, extras=extras)
                            else:
                                forward_out = gt_forward_out
                        else:
                            with timer.env('Network Extra'):
                                forward_out = net(batch, extras={"backbone": "full"})
                        preds = forward_out["pred_outs"]
                        if args.display:
                            img_numpy = prep_display(preds, img, h, w)
                        elif args.benchmark:
                            new_h, new_w = h, w
                            if new_w > 640:
                                new_w, new_h = 640, 640 * new_h // new_w
                            if new_h > 480:
                                new_h, new_w = 480, 480 * new_w // new_h
                            prep_benchmark(preds, new_h, new_w)
                        elif annot_idx != -1:
                            prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[video_idx], detections)
                        if it > 0 or pass_idx > 0 or frame_seq_idx > frame_eval_stride:
                            frame_times.add(timer.total_time())
                        if args.display:
                            if it > 0 or pass_idx > 0 or frame_seq_idx > frame_eval_stride:
                                print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                            plt.imshow(img_numpy)
                            plt.title(str(dataset.ids[video_idx]))
                            plt.show()
                        elif not args.no_bar:
                            if it > 0 or pass_idx > 0 or frame_seq_idx > frame_eval_stride: fps = 1 / frame_times.get_avg()
                            else: fps = 0
                            progress = (it+1) / dataset_size * 100
                            progress_bar.set_val(it+1)
                            print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                                  % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')
        else:
            for it, image_idx in enumerate(dataset_indices):
                timer.reset()
                with timer.env('Load Data'):
                    img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
                    batch = Variable(img.unsqueeze(0))
                    if args.cuda:
                        batch = batch.cuda()
                with timer.env('Network Extra'):
                    extras = {"backbone": "full", "interrupt": False,
                              "moving_statistics": {"aligned_feats": []}}
                    preds = net(batch, extras=extras)["pred_outs"]
                if args.display:
                    img_numpy = prep_display(preds, img, h, w)
                elif args.benchmark:
                    prep_benchmark(preds, h, w)
                else:
                    prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
                if it > 1:
                    frame_times.add(timer.total_time())
                if args.display:
                    if it > 1:
                        print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                    plt.imshow(img_numpy)
                    plt.title(str(dataset.ids[image_idx]))
                    plt.show()
                elif not args.no_bar:
                    if it > 1: fps = 1 / frame_times.get_avg()
                    else: fps = 0
                    progress = (it+1) / dataset_size * 100
                    progress_bar.set_val(it+1)
                    print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                          % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')
        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)
                calc_map(ap_data)
        elif args.benchmark:
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))
    except KeyboardInterrupt:
        if not args.display and not args.benchmark:
            print()
            logger = logging.getLogger("yolact.eval")
            logger.info('Stopping early, calculating AP based on finished proportion...')
            calc_map(ap_data)
        elif args.benchmark:
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

def calc_map(ap_data):
    logger = logging.getLogger("yolact.eval")
    logger.info('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]
    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]
                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())
    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    print_maps(all_maps)

def print_maps(all_maps):
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n: ('-------+' * n)
    output_str = "\n"
    output_str += make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]) + "\n"
    output_str += make_sep(len(all_maps['box']) + 1) + "\n"
    for iou_type in ('box', 'mask'):
        output_str += make_row([iou_type] + ['%.2f' % x for x in all_maps[iou_type].values()]) + "\n"
    output_str += make_sep(len(all_maps['box']) + 1)
    logger = logging.getLogger("yolact.eval")
    logger.info(output_str)

def badhash(x):
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x = ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

##############################################
# Main
##############################################

if __name__ == '__main__':
    parse_args()
    if args.config is not None:
        set_cfg(args.config)
    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)
    if args.detect:
        cfg.eval_mask_branch = False
    if args.dataset is not None:
        set_dataset(args.dataset)
    from yolact_edge.utils.logging_helper import setup_logger
    setup_logger(logging_level=logging.INFO)
    logger = logging.getLogger("yolact.eval")
    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')
        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            if args.deterministic:
                cudnn.deterministic = True
                cudnn.benchmark = False
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()
        if args.image is None and args.video is None and args.images is None:
            if cfg.dataset.name == 'YouTube VIS':
                dataset = YoutubeVIS(image_path=cfg.dataset.valid_images,
                                     info_file=cfg.dataset.valid_info,
                                     configs=cfg.dataset,
                                     transform=BaseTransformVideo(MEANS), has_gt=cfg.dataset.has_gt)
            else:
                dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                        transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None
        logger.info('Loading model...')
        net = Yolact(training=False)
        if args.trained_model is not None:
            net.load_weights(args.trained_model, args=args)
        else:
            logger.warning("No weights loaded!")
        net.eval()
        logger.info('Model loaded.')
        convert_to_tensorrt(net, cfg, args, transform=BaseTransform())
        if args.cuda:
            net = net.cuda()
        evaluate(net, dataset)
