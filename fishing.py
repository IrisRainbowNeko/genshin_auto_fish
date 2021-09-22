#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

from loguru import logger

import torch
import keyboard
import winsound

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info

from fisher.environment import *
from fisher.predictor import *
from fisher.models import FishNet

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./assets/dog.jpg", help="path to images or video")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    # DQN args
    parser.add_argument('--n_states', default=3, type=int)
    parser.add_argument('--n_actions', default=2, type=int)
    parser.add_argument('--step_tick', default=12, type=int)
    parser.add_argument('--model_dir', default='./weights/fish_genshin_net.pth', type=str)

    return parser

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        if args.ckpt is None:
            trt_file = os.path.join(file_name, "model_trt.pth")
        else:
            trt_file = args.ckpt
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, FISH_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)

    agent = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    agent.load_state_dict(torch.load(args.model_dir))
    agent.eval()

    print('init ok')
    if args.demo == "image":
        start_fishing(predictor, agent)

def start_fishing(predictor, agent, bite_timeout=20):
    ff = FishFind(predictor)
    env = Fishing(delay=0.1, max_step=10000, show_det=True)

    winsound.Beep(500, 500)
    keyboard.wait('r')

    while True:
        ff.do_fish()
        winsound.Beep(700, 500)
        times=0
        while True:
            if env.is_bite():
                break
            time.sleep(0.5)
            times+=1
            if times>bite_timeout:
                env.drag()
                time.sleep(3)
                ff.do_fish(fish_init=False)
                times=0
        winsound.Beep(900, 500)
        env.drag()
        time.sleep(1)

        state = env.reset()
        for i in range(env.max_step):
            state = torch.FloatTensor(state).unsqueeze(0)
            action = agent(state)
            action = torch.argmax(action, dim=1).numpy()
            state, reward, done = env.step(action)
            if done:
                break
        time.sleep(3)

#python fishing.py image -f yolox/exp/yolox_tiny_fish.py -c weights/best_tiny3.pth --conf 0.25 --nms 0.45 --tsize 640 --device gpu
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
