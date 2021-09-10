import torch
from torch import nn

class FishNet(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        layers=[
            nn.Linear(in_ch, 16),
            nn.LeakyReLU(),
            nn.Linear(16, out_ch)
        ]
        super(FishNet, self).__init__(*layers)
        self.apply(weight_init)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)