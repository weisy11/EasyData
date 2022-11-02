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
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import copy
import cv2
import numpy as np
import faiss
import pickle

from predict.predict_rec import RecPredictor

# from ppcv.engine.pipeline import Pipeline
# from ppcv.utils.logger import setup_logger
# from ppcv.core.config import ArgsParser

from utils import logger
from utils import config
from utils.get_image_list import get_image_list


class SystemPredictor(object):

    def __init__(self, config):

        self.config = config
        self.rec_predictor = RecPredictor(config)

        assert 'IndexProcess' in config.keys(), "Index config not found ... "
        self.return_k = self.config['IndexProcess']['return_k']

        index_dir = self.config["IndexProcess"]["index_dir"]
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "

        if config['IndexProcess'].get("dist_type") == "hamming":
            self.Searcher = faiss.read_index_binary(
                os.path.join(index_dir, "vector.index"))
        else:
            self.Searcher = faiss.read_index(
                os.path.join(index_dir, "vector.index"))

        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)

    def append_self(self, results, shape):
        results.append({
            "class_id": 0,
            "score": 1.0,
            "bbox": np.array([0, 0, shape[1],
                              shape[0]]),  # xmin, ymin, xmax, ymax
            "label_name": "foreground",
        })
        return results

    def predict(self, img):
        output = []
        # st1: get img for inputs
        results = [img]

        # st2: recognition process, use score_thres to ensure accuracy
        for result in results:
            preds = {}
            rec_results = self.rec_predictor.predict(result)
            scores, docs = self.Searcher.search(rec_results, self.return_k)
            print(scores)
            # just top-1 result will be returned for the final
            if self.config["IndexProcess"]["dist_type"] == "hamming":
                if scores[0][0] <= self.config["IndexProcess"][
                        "hamming_radius"]:
                    preds["rec_docs"] = [
                        self.id_map[docs[0][i]] for i in range(self.return_k)
                    ]
                    preds["rec_scores"] = scores[0]
                    output.append(preds)

            else:
                if scores[0][0] >= self.config["IndexProcess"]["score_thres"]:
                    preds["rec_docs"] = [
                        self.id_map[docs[0][i]] for i in range(self.return_k)
                    ]
                    preds["rec_scores"] = scores[0]
                    output.append(preds)

        return output


def main(config, aug_name):
    all_count = 0
    write_file = open("tmp/{}_repeat.txt".format(aug_name), "w")
    system_predictor = SystemPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = system_predictor.predict(img)
        if len(output):
            # print(output[0])
            write_file.write(image_file[8:] + "\t" +
                             output[0]['rec_docs'][1][:-2] + "\t" +
                             str(output[0]['rec_scores'][1]) + "\n")
            if output[0]['rec_scores'][1] > 0.95:
                all_count += 1
    print({str(aug_name): str(all_count)})
    return


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    #for aug_name in ["randaugment", "random_erasing", "tia_perspective", "gridmask", "tia_distort", "tia_stretch"]:
    for aug_name in ["randaugment"]:
        config["Global"]["infer_imgs"] = "dataset/{}".format(aug_name)
        config["IndexProcess"]["index_dir"] = "./augdata/all_aug"
        main(config, aug_name)
