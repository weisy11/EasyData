import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import argparse
import random
import numpy as np
import faiss
import pickle
import json
import time
import logging
from PIL import Image

parent = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(parent, '../deploy/')))

from ppcv.engine.pipeline import Pipeline
from ppcv.utils.logger import setup_logger
from ppcv.core.config import ArgsParser

from deploy.python.ppaug.utils import config
from deploy.python.ppaug.utils import logger
from deploy.python.ppaug.utils.get_image_list import get_image_list_from_label_file
from deploy.python.ppaug.gen_img import GenAug
from deploy.python.ppaug.predict.build_gallery import GalleryBuilder


class PPAug(object):

    def __init__(self, cfg):
        self.save_list = []
        config_args = config.parse_args()
        self.config = config.get_config(cfg.config, show=True)
        self.gen_num = self.config["DataGen"]["gen_num"]
        self.gen_ratio = self.config["DataGen"]["gen_ratio"]

        self.aug_file = self.config["DataGen"]["aug_file"]
        self.check_dir(self.aug_file)
        self.ori_label = self.config["DataGen"]["label_file"]
        self.aug_type = self.config["DataGen"]["ops"]

        self.compare_out = self.config["FeatureExtract"]["file_out"]
        self.check_dir(self.compare_out)
        self.feature_thresh = self.config["FeatureExtract"]["thresh"]

        self.score_thresh = self.config["BigModel"]["thresh"]
        self.big_model_out = self.config["BigModel"]["final_label"]
        self.model_type = self.config["BigModel"]["model_type"]
        if not os.path.exists("tmp"):
            os.makedirs("tmp")

    def build_big_model(self):
        parser = argparse.ArgumentParser()
        parser.set_defaults(config=self.config["BigModel"]["config"])
        return parser

    def build_feature_compare(self):

        parser = argparse.ArgumentParser()
        parser.set_defaults(config=self.config["FeatureExtract"]["config"])

        assert 'IndexProcess' in self.config.keys(
        ), "Index config not found ... "
        self.return_k = self.config['IndexProcess']['return_k']

        index_dir = self.config["IndexProcess"]["index_dir"]
        assert os.path.exists(os.path.join(
            index_dir, "vector.index")), "vector.index not found ..."
        assert os.path.exists(os.path.join(
            index_dir, "id_map.pkl")), "id_map.pkl not found ... "

        if self.config['IndexProcess'].get("dist_type") == "hamming":
            self.Searcher = faiss.read_index_binary(
                os.path.join(index_dir, "vector.index"))
        else:
            self.Searcher = faiss.read_index(
                os.path.join(index_dir, "vector.index"))

        with open(os.path.join(index_dir, "id_map.pkl"), "rb") as fd:
            self.id_map = pickle.load(fd)
        return parser

    def get_label(self, data_file, delimiter=" "):
        self.all_label = {}
        with open(data_file, "r") as f:
            for line in f.readlines():
                path, label = line.strip().split(delimiter)
                path = path.split("/")[-1]
                self.all_label[path] = label
        return self.all_label

    def rm_repeat(self, compare_file, out_file, thresh):
        count = 0
        with open(out_file, "w", encoding="utf-8") as new_aug_file:
            with open(compare_file, "r") as f:
                for line in f.readlines():
                    query, gallery, score = line.strip().split("\t")
                    path = query.split("/")[-1]
                    if float(score) > thresh and (gallery or
                                                  query) not in self.save_list:
                        count += 1
                        self.save_list.append(gallery)
                        self.save_list.append(query)
                        new_aug_file.write(query + " " +
                                           str(self.all_label[path]) + "\n")
                    elif float(score) < thresh:
                        count += 1
                        self.save_list.append(query)
                        new_aug_file.write(query + " " +
                                           str(self.all_label[path]) + "\n")
        return count

    def check_dir(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        return

    def run(self):
        # gen aug data
        with open(self.aug_file, "w") as f:
            for aug_type in self.aug_type:
                self.config["DataGen"]["aug"] = aug_type
                dataaug = GenAug(self.config)
                dataaug(gen_num=self.gen_num, trans_label=f)
        # build gallery
        parser = self.build_feature_compare()
        self.feature_extract = Pipeline(parser.parse_args([]))

        GalleryBuilder(self.config, self.feature_extract)
        # feather compare
        root_path = self.config["IndexProcess"]["image_root"]
        image_list, gt = get_image_list_from_label_file(self.aug_file)

        with open("tmp/repeat.txt", "w") as write_file:
            for idx, image_file in enumerate(image_list):
                preds = {}
                output = []
                rec_results = self.feature_extract.run(
                    os.path.join(root_path, image_file))
                feature = np.array([rec_results[0]["feature"]])
                scores, docs = self.Searcher.search(feature, self.return_k)

                if scores[0][0] >= self.config["IndexProcess"]["score_thres"]:
                    preds["rec_docs"] = [
                        self.id_map[docs[0][i]] for i in range(self.return_k)
                    ]
                    preds["rec_scores"] = scores[0]
                    output.append(preds)
                if len(output):
                    write_file.write(image_file + "\t" +
                                     output[0]['rec_docs'][1][:-2] + "\t" +
                                     str(output[0]['rec_scores'][1]) + "\n")

        # rm repeat
        all_label = self.get_label(self.aug_file)
        final_count = self.rm_repeat(compare_file="tmp/repeat.txt",
                                     out_file=self.compare_out,
                                     thresh=self.feature_thresh)

        # filter low score data
        image_list, gt_labels = get_image_list_from_label_file(
            self.compare_out)
        batch_names = []

        big_parser = self.build_big_model()
        big_model = Pipeline(big_parser.parse_args([]))
        cnt = 0

        with open(self.big_model_out, "w") as save_file:
            for idx, img_path in enumerate(image_list):
                file_name = os.path.join(root_path, img_path)
                if os.path.exists(file_name):
                    batch_names.append(file_name)
                    cnt += 1
                else:
                    logger.warning(
                        "Image file failed to read and has been skipped. The path: {}"
                        .format(img_path))

                if cnt % self.config["BigModel"]["batch_size"] == 0 or (
                        idx + 1) == len(image_list):
                    if len(batch_names) == 0:
                        continue
                    # big model predict
                    batch_results = big_model.predict_images(batch_names)

                    for number, result_dict in enumerate(batch_results):
                        if self.model_type == "cls":
                            filename = batch_names[number]
                            scores_str = "[{}]".format(", ".join(
                                "{:.2f}".format(r)
                                for r in result_dict["scores"]))
                            if float(scores_str[1:-1]) > self.score_thresh:
                                save_file.write("{} {}\n".format(
                                    filename, gt_labels[idx]))
                        elif self.model_type == "ocr_rec":
                            filename = batch_names[number]
                            scores = result_dict[1]
                            if scores > self.score_thresh:
                                save_file.write("{} {}\n".format(
                                    filename, gt_labels[idx]))
