from utils.render import *
from fisher.models import FishNet
from fisher.environment import *
import torch
import argparse
from matplotlib.animation import FFMpegWriter

parser = argparse.ArgumentParser(description='Test Genshin finsing with DQN')
parser.add_argument('--n_states', default=3, type=int)
parser.add_argument('--n_actions', default=2, type=int)
parser.add_argument('--step_tick', default=12, type=int)
parser.add_argument('--model_dir', default='./output/fish_net_399.pth', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    writer = FFMpegWriter(fps=60)
    render = PltRender(call_back=lambda: writer.grab_frame())

    net = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    env = Fishing_sim(step_tick=args.step_tick, drawer=render, stop_tick=10000)

    net.load_state_dict(torch.load(args.model_dir))

    net.eval()
    state = env.reset()
    with writer.saving(render.fig, 'out.mp4', 100):
        for i in range(2000):
            env.render()

            state = torch.FloatTensor(state).unsqueeze(0)
            action = net(state)
            action = torch.argmax(action, dim=1).numpy()
            state, reward, done = env.step(action)
            if done:
                break