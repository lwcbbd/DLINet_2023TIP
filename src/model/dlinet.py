import torch
import torch.nn as nn
from torch.nn import functional as F
from model import common
from model import lmrd
from model import lmrd_tjt
from model import prre
from model import da

def make_model(args, mode):
    return DLINET(args, mode)

class DLINET(nn.Module):
    def __init__(self, args, mode):
        super(DLINET, self).__init__()

        self.mode = mode
        if self.mode == 'first_stage':
            self.rain_detect = lmrd_tjt.RAIN_DETECTION(common.ResBlock,args.num_mask)
        elif self.mode == 'second_stage':
            self.rain_detect = lmrd.RAIN_DETECTION(common.ResBlock, args.num_mask)
        else:
            print("mode error!!!")

        self.rain_estimate = prre.RAIN_ESTIMATION(common.ResBlock,common.MaskSpatialAttention,args.num_msablock, args.num_mask)

        self.fusion = da.FUSION(args.num_msablock,args.num_mask)

    def forward(self, input):
        if self.mode == 'first_stage':
            x, y = input[0], input[1]
            masks = self.rain_detect(x, y)
            streaks = self.rain_estimate(x, masks)
            out = self.fusion(streaks, masks)
            return out
        elif self.mode == 'second_stage':
            x = input[0]
            masks = self.rain_detect(x)
            streaks = self.rain_estimate(x, masks)
            out = self.fusion(streaks, masks)
            return out, masks


