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
import sys

parent = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(parent, './deploy/')))

import argparse

from ppcv.engine.pipeline import Pipeline
from tools.predict_aug import PPAug
from deploy.python.ppaug.utils import config

__all__ = ['EasyData']

VERSION = '0.5.0.1'

CONFIG_URLS = {
    'PPEDA': {
        'config': "./deploy/configs/ppeda.yaml"
    },
    'PPLDI': {
        'config': "./deploy/configs/ppidl.yaml"
    }
}


def argsparser():

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=
        "Path of input, suport image file, image directory and video file.",
        required=False)
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Directory of output visualization files.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default=None,
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=
        "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )

    args = parser.parse_args()
    return vars(args)


def init_config(**cfg):
    model_name = cfg["model"]
    base_cfg_path = f"./deploy/configs/{model_name}.yaml"
    __dir__ = os.path.dirname(__file__)
    base_cfg_path = os.path.join(__dir__, base_cfg_path)

    env_config = {}
    if "output_dir" in cfg and cfg["output_dir"]:
        env_config["output_dir"] = cfg["output_dir"]
    if "run_mode" in cfg and cfg["run_mode"]:
        env_config["run_mode"] = cfg["run_mode"]
    if "device" in cfg and cfg["device"]:
        env_config["device"] = cfg["device"]
    opt_config = env_config

    FLAGS = argparse.Namespace(**{"config": base_cfg_path, "opt": opt_config})

    return FLAGS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ppeda",
    )
    parser.add_argument("--gen_num", type=int, default=10)
    parser.add_argument("--gen_ratio", type=float, default=0)
    parser.add_argument("--ops",
                        type=list,
                        default=[
                            "randaugment", "random_erasing", "gridmask",
                            "tia_distort", "tia_stretch", "tia_perspective"
                        ])
    parser.add_argument("--ori_data_dir",
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument("--label_file", type=str, default=None, required=True)
    parser.add_argument("--aug_file", type=str, default="labels/test.txt")
    parser.add_argument("--out_dir", type=str, default="test")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--repeat_ratio", type=float, default=0.9)
    parser.add_argument("--compare_out", type=str, default="rm_repeat.txt")
    parser.add_argument("--use_big_model", type=bool, default=True)
    parser.add_argument("--quality_ratio", type=float, default=0.4)
    parser.add_argument("--final_label",
                        type=str,
                        default="high_socre_label.txt")
    parser.add_argument("--model_type", type=str, default="cls")
    return parser.parse_args()


class PPEDA(PPAug):

    def __init__(self, **kwargs):
        args = parse_args()
        self.save_list = []
        model_config = CONFIG_URLS['PPEDA']['config']
        self.config = config.get_config(model_config, show=True)

        self.gen_num = args.gen_num
        self.gen_ratio = args.gen_ratio

        self.ori_label = args.label_file
        self.aug_file = args.aug_file
        self.check_dir(self.aug_file)
        self.aug_type = args.ops

        self.feature_thresh = args.repeat_ratio
        self.compare_out = args.compare_out
        self.score_thresh = args.quality_ratio
        self.use_big_model = args.use_big_model
        if not args.use_big_model:
            self.compare_out = args.final_label
        else:
            self.big_model_out = args.final_label
            self.model_type = args.model_type

    def predict(self):
        self.run()


class EasyData(object):

    def __init__(self, **cfg):
        self.model = cfg['model']
        if self.model == "ppeda":
            self.pipeline = PPEDA()
        elif self.model == "ppldi":
            FLAGS = init_config(**cfg)
            self.pipeline = Pipeline(FLAGS)

    def predict(self, input=None):
        if self.model == "ppeda":
            return self.pipeline.predict()
        elif self.model == "ppldi":
            return self.pipeline.run(input)


# for CLI
def main():
    args = parse_args()
    easydata = EasyData(**(args.__dict__))
    easydata.predict()


if __name__ == "__main__":
    main()
