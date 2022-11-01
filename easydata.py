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
        required=True)
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
    base_cfg_path = f"./deploy/configs/{model_name}.yml"
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


class EasyData(object):
    def __init__(self, **cfg):
        FLAGS = init_config(**cfg)
        self.pipeline = Pipeline(FLAGS)

    def predict(self, input):
        return self.pipeline.run(input)


# for CLI
def main():
    cfg = argsparser()
    easydata = EasyData(**cfg)
    input = os.path.abspath(cfg["input"])
    easydata.predict(input)


if __name__ == "__main__":
    main()
