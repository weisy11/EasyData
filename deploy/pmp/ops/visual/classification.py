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

logger = setup_logger('ClasVisualizer')


@register
class ClasVisualizer(Visualizer):
    def __init__(self, env_cfg, model_cfg):
        super(ClasVisualizer, self).__init__(env_cfg, model_cfg)

    def __call__(self, inputs, vis=False):
        output = []
        for input in inputs:
            fn, image, class_ids, scores, label_names = input
            res = dict(
                filename=fn,
                class_ids=class_ids,
                scores=scores,
                label_names=label_names)
            if self.frame_id != -1:
                res.update({'frame_id': frame_id})
            logger.info(res)
            if vis:
                image = image[:, :, ::-1]
                output.append({'output': image})
        return output
