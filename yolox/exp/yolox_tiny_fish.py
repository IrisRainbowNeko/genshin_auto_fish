#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_scale = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

        # Define yourself dataset path
        self.data_dir = "../datas/fish_dataset"
        self.data_name = 'images'
        self.val_name = 'images'
        self.train_ann = "fish_p2_75+jam.json"
        self.val_ann = "fish_p2_75+jam.json"

        self.num_classes = 6

        self.max_epoch = 35
        self.data_num_workers = 4
        self.eval_interval = 10

        self.basic_lr_per_img = 2e-3 / 8.0
