# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import os
import numpy as np
import math
import glob
import paddle
import cv2
from collections import defaultdict
from pmp.ops.visual.base import Visualizer
from pmp.utils.logger import setup_logger
from pmp.core.workspace import register
from PIL import Image, ImageDraw, ImageFile

logger = setup_logger('DetVisualizer')


def get_color_map_list(num_classes):
    """
    Args:
        num_classes (int): number of class
    Returns:
        color_map (list): RGB color list
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_det(image, dt_bboxes, dt_scores, dt_cls_names):
    im = Image.fromarray(image[:, :, ::-1])
    draw_thickness = min(im.size) // 320
    draw = ImageDraw.Draw(im)
    name_set = set(dt_cls_names)
    name2clsid = {name: i for i, name in enumerate(name_set)}
    clsid2color = {}
    color_list = get_color_map_list(len(name_set))

    for box, score, name in zip(dt_bboxes, dt_scores, dt_cls_names):
        color = tuple(color_list[name2clsid[name]])

        xmin, ymin, xmax, ymax = box
        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=draw_thickness,
            fill=color)

        # draw label
        text = "{} {:.4f}".format(name, score)
        box = draw.textbbox((xmin, ymin), text, anchor='lt')
        draw.rectangle(box, fill=color)
        draw.text((box[0], box[1]), text, fill=(255, 255, 255))
    image = np.array(im)
    return image


@register
class DetVisualizer(Visualizer):
    def __init__(self, env_cfg, model_cfg):
        super(DetVisualizer, self).__init__(env_cfg, model_cfg)

    def __call__(self, inputs, vis=False):
        output = []
        for input in inputs:
            fn, image, dt_bboxes, dt_scores, dt_cls_names = input
            res = dict(
                filename=fn,
                dt_bboxes=dt_bboxes,
                dt_scores=dt_scores,
                dt_cls_names=dt_cls_names)
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if vis:
                vis_img = draw_det(image, dt_bboxes, dt_scores, dt_cls_names)
                output.append({'output': vis_img})
        return output
