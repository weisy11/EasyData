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
import paddle
from collections import defaultdict
from collections.abc import Sequence
import yaml
import copy
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pmp.utils.logger import setup_logger
import pmp
from pmp.ops import *

logger = setup_logger('config')


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


class ConfigParser(object):
    def __init__(self, args):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        self.env_cfg, self.model_cfg = self.merge_cfg(args, cfg)
        self.check_cfg()

    def merge_cfg(self, args, cfg):
        env_cfg = cfg['ENV']
        model_cfg = cfg['MODEL']

        def merge(cfg, arg):
            merge_cfg = copy.deepcopy(cfg)
            for k, v in env_cfg.items():
                if k in arg:
                    merge_cfg[k] = arg[k]
                else:
                    if isinstance(v, dict):
                        merge_cfg[k] = merge(v, arg)
            return merge_cfg

        def merge_opt(cfg, arg):
            merge_cfg = copy.deepcopy(cfg)
            # merge opt
            if 'opt' in arg.keys() and arg['opt']:
                if 'ENV' in arg['opt'] or 'MODEL' in arg['opt']:
                    raise ValueError('-o do not need ENV or MODEL field.')
                for name, value in arg['opt'].items(
                ):  # example: {'MOT': {'batch_size': 3}}
                    if name not in merge_cfg.keys():
                        print("No", name, "in config file!")
                        continue
                    for sub_k, sub_v in value.items():
                        if sub_k not in merge_cfg[name].keys():
                            print("No", sub_k, "in config file of", name, "!")
                            continue
                        merge_cfg[name][sub_k] = sub_v

            return merge_cfg

        args_dict = vars(args)
        env_cfg = merge(env_cfg, args_dict)
        env_cfg = merge_opt(env_cfg, args_dict)
        model_cfg = merge_opt(model_cfg, args_dict)
        return env_cfg, model_cfg

    def check_cfg(self):
        unique_name = set()
        unique_name.add('input')
        op_list = pmp.ops.__all__
        output_set = {'input.image', 'input.video'}
        for model in self.model_cfg:
            model_name = list(model.keys())[0]
            model_dict = list(model.values())[0]

            # check the name and last_ops is legal
            if 'name' not in model_dict or 'last_ops' not in model_dict:
                raise ValueError(
                    'Missing name or last_op field in {} model config'.format(
                        model_name))

            last_ops = model_dict['last_ops']
            assert isinstance(
                last_ops, Sequence
            ), 'The last_ops must be sequence, but the type in {} is {}'.format(
                model_name, type(last_ops))
            for last_op in last_ops:
                assert last_op in unique_name, 'The last_op {} in {} model config is not exist.'.format(
                    model_dict['last_ops'], model_name)
            unique_name.add(model_dict['name'])

        device = self.env_cfg['device']
        assert device.upper() in ['CPU', 'GPU', 'XPU'
                                  ], "device should be CPU, GPU or XPU"

    def get_input_bs(self):
        for model in self.model_cfg:
            model_dict = list(model.values())[0]
            #model_dict['last_ops'] == ['input']
            return model_dict.get('batch_size', 1)

    def parse(self):
        return self.env_cfg, self.model_cfg

    def print_cfg(self):
        print('----------- Environment Arguments -----------')
        buffer = yaml.dump(self.env_cfg)
        print(buffer)
        print('------------- Model Arguments ---------------')
        buffer = yaml.dump(self.model_cfg)
        print(buffer)
        print('---------------------------------------------')
