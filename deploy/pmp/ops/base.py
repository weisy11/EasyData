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
import importlib
import math
import numpy as np
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from pmp.ops.predictor import PaddlePredictor
from pmp.utils.download import get_model_path

__all__ = [
    "BaseOp",
]


def create_operators(params, mod):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
        mod(module) : a module that can import single ops
    """
    assert isinstance(params, list), ('operator config should be a list')
    if mod is None:
        mod = importlib.import_module(__name__)
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(mod, op_name)(**param)
        ops.append(op)

    return ops


class BaseOp(object):
    """
    Base Operator, implement of prediction process
    Args
    """

    def __init__(self, model_cfg, env_cfg):
        param_path = get_model_path(model_cfg['param_path'])
        model_path = get_model_path(model_cfg['model_path'])
        env_cfg["batch_size"] = model_cfg.get("batch_size", 1)
        self.batch_size = env_cfg["batch_size"]
        self.name = model_cfg["name"]
        self.frame = -1
        self.predictor = PaddlePredictor(param_path, model_path, env_cfg)

        self.input_keys = model_cfg["Inputs"]
        keys = self.get_output_keys()
        self.output_keys = [self.name + '.' + key for key in keys]

    @classmethod
    def type(self):
        return 'MODEL'

    @classmethod
    def get_output_keys(cls):
        raise NotImplementedError

    def get_input_keys(self):
        return self.input_keys

    def filter_input(self, last_outputs, input_name):
        f_inputs = []
        for output in last_outputs:
            f_input = [output[k] for k in input_name]
            f_inputs.append(f_input)

        return f_inputs

    def set_frame(self, frame_id):
        self.frame_id = frame_id

    def preprocess(self, inputs):
        raise NotImplementedError

    def postprocess(self, inputs):
        raise NotImplementedError

    def run(self, image_list):
        raise NotImplementedError

    def merge_batch_result(self, batch_result):
        raise NotImplementedError
