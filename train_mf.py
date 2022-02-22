from fisher.agent import DQN
from fisher.models import FishNet, MoveFishNet
from fisher.environment import *
import torch
import argparse
import os
import keyboard
import winsound
from loguru import logger
from fisher.predictor import *
from yolox.exp import get_exp

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
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_states', default=8, type=int)
    parser.add_argument('--n_actions', default=3, type=int)
    parser.add_argument('--n_episode', default=400, type=int)
    parser.add_argument('--save_dir', default='./output', type=str)
    parser.add_argument('--resume', default=None, type=str)

    return parser

def get_predictor(exp, args):
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

    return Predictor(model, exp, FISH_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)

args = make_parser().parse_args()
exp = get_exp(args.exp_file, args.name)

predictor = get_predictor(exp, args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

net = MoveFishNet(in_ch=args.n_states, out_ch=args.n_actions)
if args.resume:
    net.load_state_dict(torch.load(args.resume))

agent = DQN(net, args.batch_size, args.n_states, args.n_actions, memory_capacity=1000, reg=True)
env = FishMove(predictor)

#python train_mf.py image -f yolox/exp/yolox_tiny_fish.py -c weights/best_tiny3.pth --conf 0.25 --nms 0.45 --tsize 640 --device gpu
if __name__ == '__main__':
    # Start training
    print("\nCollecting experience...")
    net.train()
    for i_episode in range(args.n_episode):
        winsound.Beep(500, 500)
        keyboard.wait('r')
        # play 400 episodes of cartpole game
        s = env.reset()
        ep_r = 0
        while True:
            # take action based on the current state
            a = agent.choose_action(s)
            # obtain the reward and next state and some other information
            s_, r, done = env.step(a)

            # store the transitions of states
            agent.store_transition(s, a, r, s_, int(done))

            ep_r += r
            # if the experience repaly buffer is filled, DQN begins to learn or update
            # its parameters.
            if agent.memory_counter > agent.memory_capacity:
                agent.train_step()
                if done:
                    print('Ep: ', i_episode, ' |', 'Ep_r: ', round(ep_r, 2))

            if done:
                # if game is over, then skip the while loop.
                break
            # use next state to update the current state.
            s = s_
        torch.save(net.state_dict(), os.path.join(args.save_dir, f'fish_move_net_{i_episode}.pth'))