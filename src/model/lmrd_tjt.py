import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model import common

class RAIN_DETECTION(nn.Module):
    def __init__(self, resblock,num_mask):
        super(RAIN_DETECTION, self).__init__()
        self.input_encoder_rain = nn.Sequential(
            nn.Conv2d(3, 32, bias=True, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')
        )
        self.input_encoder_norain = nn.Sequential(
            nn.Conv2d(3, 32, bias=True, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')
        )

        self.body_layer = self.make_layer(resblock, 8, 32)

        self.mask_decoder = nn.Sequential(
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(32, num_mask, bias=True, kernel_size=3, padding='same')
        )

        self.act = common.ActBinarizer_01()

    def make_layer(self, resblock, num_resblock, n_feats):
        layers = []
        for _ in range(0, num_resblock):
            layers.append(resblock(n_feats, bias=True, bn=True, act=nn.ReLU(True), res_scale=1))
        return nn.Sequential(*layers)

    def forward(self, rain, norain):
        rain_f = self.input_encoder_rain(rain)
        norain_f = self.input_encoder_norain(norain)
        f = self.body_layer(rain_f+norain_f)
        masks = self.mask_decoder(f)
        out = self.act(masks)
        return out
