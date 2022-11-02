# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import copy
import paddle
import numpy as np
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
# from ppcls.utils import logger

from python.ppaug.data import preprocess
from python.ppaug.data.preprocess import transform


def create_operators(params, class_num=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op_func = getattr(preprocess, op_name)
        if "class_num" in inspect.getfullargspec(op_func).args:
            param.update({"class_num": class_num})
        op = op_func(**param)
        ops.append(op)

    return ops
