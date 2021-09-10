from render import *
from models import FishNet
from environment import *
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='Test Genshin finsing with DQN')
parser.add_argument('--n_states', default=3, type=int)
parser.add_argument('--n_actions', default=2, type=int)
parser.add_argument('--step_tick', default=12, type=int)
parser.add_argument('--model_dir', default='./output/fish_net_399.pth', type=str)
args = parser.parse_args()

if __name__ == '__main__':

    net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    env = Fishing(delay=0.1)

    net.load_state_dict(torch.load(args.model_dir))

    net.eval()
    state = env.step(0)[0]
    for i in range(2000):
        env.render()

        state = torch.FloatTensor(state).unsqueeze(0)
        action = net(state)
        action = torch.argmax(action, dim=1).numpy()
        state, reward, done = env.step(action)
        if done:
            break