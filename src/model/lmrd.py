import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model import common

class RAIN_DETECTION(nn.Module):
    def __init__(self, resblock,num_mask):
        super(RAIN_DETECTION, self).__init__()
        self.num_mask = num_mask

        self.rain_encoder = nn.Sequential(
            nn.Conv2d(3, 32, bias=True, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')
        )

        self.multi_scale_module = HRNet(resblock)
        self.laplacian_module = LPF_Net(resblock)

        self.mask_decoder_s = nn.Sequential(
            nn.Conv2d(32*3, 32, bias=True, kernel_size=3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(32, num_mask, bias=True, kernel_size=3, padding='same')
        )

        self.act = F.sigmoid

    def forward(self,x):
        x_f = self.rain_encoder(x)
        multi_scale_f = self.multi_scale_module(x_f)
        laplacian_f = self.laplacian_module(multi_scale_f)
        masks = self.mask_decoder_s(torch.cat(laplacian_f,1))
        out = self.act(masks)
        return out

class HRNetbranch(nn.Module):
    def __init__(self, block, num_block, num_feats):
        super(HRNetbranch, self).__init__()
        self.num_block = num_block
        self.num_feats = num_feats

        self.body = self.make_layer(block, self.num_block, self.num_feats)
        self.act = F.relu

    def make_layer(self, resblock, num_resblock, n_feats):
        layers = []
        for _ in range(0, num_resblock):
            layers.append(resblock(n_feats, bias=False, bn=True, act=nn.ReLU(True), res_scale=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        out = self.act(out)
        return out


class HRNetbrachfuse(nn.Module):
    def __init__(self, inchannellist, outinchannellist):
        super(HRNetbrachfuse, self).__init__()
        self.inchannellist = inchannellist
        self.inbranchnum = len(inchannellist)
        self.outinchannellist = outinchannellist
        self.outbranchnum = len(outinchannellist)

        fuse_layer_list = []
        for i in range(self.outbranchnum):
            fuse_layers = []
            for j in range(self.inbranchnum):
                if j == i:
                    fuse_layers.append(None)

                elif j > i:
                    fuse_layers.append(
                        nn.Sequential(
                            nn.Conv2d(inchannellist[j], outinchannellist[i], bias=False, kernel_size=1, padding='same'),
                            nn.BatchNorm2d(outinchannellist[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )

                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(inchannellist[j], outinchannellist[i], bias=False, kernel_size=3,
                                              stride=2, padding=1),
                                    nn.BatchNorm2d(outinchannellist[i])
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(inchannellist[j], inchannellist[j], bias=False, kernel_size=3, stride=2,
                                              padding=1),
                                    nn.BatchNorm2d(inchannellist[j]),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layers.append(nn.Sequential(*conv3x3s))
            fuse_layer_list.append(nn.ModuleList(fuse_layers))

        self.fuse_layer_list = nn.ModuleList(fuse_layer_list)

    def forward(self, xlist):
        if len(xlist) != self.inbranchnum:
            print("input channel error!")

        fuse_list = []
        for i in range(self.outbranchnum):
            y = xlist[0] if i == 0 else self.fuse_layer_list[i][0](xlist[0])
            for j in range(1, len(xlist)):
                if i == j:
                    y = y + xlist[j]
                else:
                    y = y + self.fuse_layer_list[i][j](xlist[j])
            fuse_list.append(F.relu(y))

        if len(fuse_list) != self.outbranchnum:
            print("output channel error!")

        return fuse_list


class HRNet(nn.Module):
    def __init__(self, resblock):
        super(HRNet, self).__init__()

        self.stage1_branch1 = HRNetbranch(resblock, num_feats=32, num_block=3)

        self.stage2fuse = HRNetbrachfuse([32], [32, 32])
        self.stage2_branch1 = HRNetbranch(resblock, num_feats=32, num_block=3)
        self.stage2_branch2 = HRNetbranch(resblock, num_feats=32, num_block=3)

        self.stage3fuse = HRNetbrachfuse([32, 32], [32, 32, 32])
        self.stage3_branch1 = HRNetbranch(resblock, num_feats=32, num_block=3)
        self.stage3_branch2 = HRNetbranch(resblock, num_feats=32, num_block=3)
        self.stage3_branch3 = HRNetbranch(resblock, num_feats=32, num_block=3)

        self.stage4fuse = HRNetbrachfuse([32, 32, 32], [32, 32, 32, 32])
        self.stage4_branch1 = HRNetbranch(resblock, num_feats=32, num_block=3)
        self.stage4_branch2 = HRNetbranch(resblock, num_feats=32, num_block=3)
        self.stage4_branch3 = HRNetbranch(resblock, num_feats=32, num_block=3)
        self.stage4_branch4 = HRNetbranch(resblock, num_feats=32, num_block=3)

        self.outfuse = HRNetbrachfuse([32, 32, 32, 32], [32, 32, 32, 32])

    def forward(self, x):
        stage1_branch1 = self.stage1_branch1(x)

        stage2_fuse = self.stage2fuse([stage1_branch1])
        stage2_branch1 = self.stage2_branch1(stage2_fuse[0])
        stage2_branch2 = self.stage2_branch2(stage2_fuse[1])

        stage3_fuse = self.stage3fuse([stage2_branch1, stage2_branch2])
        stage3_branch1 = self.stage3_branch1(stage3_fuse[0])
        stage3_branch2 = self.stage3_branch2(stage3_fuse[1])
        stage3_branch3 = self.stage3_branch3(stage3_fuse[2])

        stage4_fuse = self.stage4fuse([stage3_branch1, stage3_branch2, stage3_branch3])
        stage4_branch1 = self.stage4_branch1(stage4_fuse[0])
        stage4_branch2 = self.stage4_branch2(stage4_fuse[1])
        stage4_branch3 = self.stage4_branch3(stage4_fuse[2])
        stage4_branch4 = self.stage4_branch4(stage4_fuse[3])

        out = self.outfuse([stage4_branch1, stage4_branch2, stage4_branch3, stage4_branch4])
        return out


class LAPLACIAN_BLOCK(nn.Module):
    def __init__(self, resblock):
        super(LAPLACIAN_BLOCK, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.x_layer = self.make_layer(resblock, 4, 32)
        self.x_down_layer = self.make_layer(resblock, 4, 32)
        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')
        )

        self.act = nn.ReLU()

    def make_layer(self, resblock, num_resblock, n_feats):
        layers = []
        for _ in range(0, num_resblock):
            layers.append(resblock(n_feats, bias=True, bn=True, act=nn.ReLU(True), res_scale=1))
        return nn.Sequential(*layers)

    def forward(self, x, down_x, up_shape):
        down_x_f = self.act(self.x_down_layer(down_x))
        x_f = self.act(self.x_layer(x))
        down_ip_f = F.upsample(down_x_f, (up_shape[0], up_shape[1]))
        x_ip_f = F.upsample(x_f, (up_shape[0], up_shape[1]))
        res_f = x_ip_f - down_ip_f
        mask = self.out_layer(res_f)
        return mask, down_x_f


class LPF_Net(nn.Module):
    def __init__(self, resblock):
        super(LPF_Net, self).__init__()
        self.lpf_1 = LAPLACIAN_BLOCK(resblock)
        self.lpf_2 = LAPLACIAN_BLOCK(resblock)
        self.lpf_3 = LAPLACIAN_BLOCK(resblock)

        self.fuse1 = nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')
        self.fuse2 = nn.Conv2d(32, 32, bias=True, kernel_size=3, padding='same')

    def make_layer(self, resblock, num_resblock, n_feats):
        layers = []
        for _ in range(0, num_resblock):
            layers.append(resblock(n_feats, bias=True, bn=True, act=nn.ReLU(True), res_scale=1))
        return nn.Sequential(*layers)

    def forward(self, xlist):
        level_o = xlist[0]
        img_h_levelo = level_o.shape[2]
        img_w_levelo = level_o.shape[3]

        level_1 = xlist[1]
        img_h_level1 = level_1.shape[2]
        img_w_level1 = level_1.shape[3]

        level_2 = xlist[2]
        img_h_level2 = level_2.shape[2]
        img_w_level2 = level_2.shape[3]

        level_3 = xlist[3]

        laplace1, up_2 = self.lpf_1(level_o, level_1, (img_h_levelo, img_w_levelo))
        laplace2, up_3 = self.lpf_2(up_2, level_2, (img_h_level1, img_w_level1))
        laplace3, up_4 = self.lpf_3(up_3, level_3, (img_h_level2, img_w_level2))

        laplace2_up = F.upsample(laplace2, (img_h_levelo, img_w_levelo))
        laplace1_2 = self.fuse1(laplace2_up + laplace1)

        laplace3_up = F.upsample(laplace3, (img_h_levelo, img_w_levelo))
        laplace1_3 = self.fuse2(laplace3_up + laplace1)

        return [laplace1, laplace1_2, laplace1_3]