import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model import common

class MSA_BLOCK(nn.Module):
    def __init__(self, resblock, sa_block, num_mask):
        super(MSA_BLOCK, self).__init__()
        self.rain_encoder = nn.Conv2d(3, num_mask, bias=True, kernel_size=3, padding='same')
        self.sa_block = sa_block()
        self.basic_block = BASIC_BLOCK(resblock, num_resblock=4, n_feats=32)
        self.decoder = nn.Conv2d(32, 3, bias=True, kernel_size=3, padding='same')
        self.act = nn.LeakyReLU(0.2)

    def forward(self, b, f, m):
        sa = self.sa_block(self.rain_encoder(b),m)
        fr = self.basic_block(self.act(f)) * sa
        steak = self.decoder(self.act(fr))
        f_next = f - fr
        b_next = b - steak
        return steak, f_next, b_next

class RAIN_ESTIMATION(nn.Module):
    def __init__(self, resblock, sa_block, num_block, num_mask):
        super(RAIN_ESTIMATION, self).__init__()
        self.num_block = num_block

        self.rain_encoder = nn.Sequential(
            nn.Conv2d(3, 32, bias=True, kernel_size=3, padding='same'),
            # nn.ReLU(),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')
        )

        msa_list = []
        for i in range(num_block):
            msa_list.append(MSA_BLOCK(resblock, sa_block, num_mask))
        self.msa_list = nn.ModuleList(msa_list)

    def forward(self, x, mask):
        streak_list = []

        f0 = self.rain_encoder(x)
        b0 = x

        f = f0
        b = b0
        for i in range(self.num_block):
            streak, f, b = self.msa_list[i](b, f, mask)
            streak_list.append(streak)

        out = streak_list
        return out


class BASIC_BLOCK(nn.Module):
    def __init__(self, resblock, num_resblock, n_feats):
        super(BASIC_BLOCK, self).__init__()

        self.bodylayer = self.make_layer(resblock, num_resblock, n_feats)
        # self.act = nn.ReLU(True)
        self.act = nn.LeakyReLU(0.2)

    def make_layer(self, resblock, num_resblock, n_feats):
        layers = []
        for _ in range(0, num_resblock - 1):
            layers.append(resblock(n_feats, bias=True, bn=True, act=nn.LeakyReLU(0.2), res_scale=1))
            layers.append(nn.LeakyReLU(0.2))
        layers.append(resblock(n_feats, bias=True, bn=True, act=nn.LeakyReLU(0.2), res_scale=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        body = self.bodylayer(x)
        out = body + x
        return out