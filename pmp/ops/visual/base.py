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


class Visualizer(object):
    def __init__(self, env_cfg, model_cfg):
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.input_keys = model_cfg["Inputs"]

    @classmethod
    def type(self):
        return 'OUTPUT'

    def get_input_keys(self):
        return self.input_keys

    def set_frame(self, frame_id):
        self.frame_id = frame_id

    def filter_input(self, last_outputs, input_name):
        f_inputs = []
        for output in last_outputs:
            f_input = [output[k] for k in input_name]
            f_inputs.append(f_input)
        return f_inputs

    def __call__(self, input, vis=False):
        image = input[0][0]
        return [{'output': image}]
