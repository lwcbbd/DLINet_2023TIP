import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model import common

class FUSION_BLOCK(nn.Module):
    def __init__(self,num_mask):
        super(FUSION_BLOCK, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.pos_attention1 = nn.Sequential(
            nn.Conv2d(num_mask, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, num_mask, kernel_size=1, bias=True)
        )
        self.pos_attention2 = nn.Sequential(
            nn.Conv2d(num_mask, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, num_mask, kernel_size=1, bias=True)
        )
        self.neg_attention1 = nn.Sequential(
            nn.Conv2d(num_mask, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, num_mask, kernel_size=1, bias=True)
        )
        self.neg_attention2 = nn.Sequential(
            nn.Conv2d(num_mask, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, num_mask, kernel_size=1, bias=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, masks, streak):
        streak_1c = streak[:,0,:,:] + streak[:,1,:,:] + streak[:,2,:,:]
        streak_1c = streak_1c.unsqueeze(1)

        posfusion_10 = masks*streak_1c
        pos_avg_at = self.pos_attention1(self.avg_pool(posfusion_10))
        pos_max_at = self.pos_attention2(self.max_pool(posfusion_10))
        pos_at = self.sigmoid(pos_avg_at + pos_max_at)

        negfusion_10 = (1-masks)*streak_1c
        neg_avg_at = self.neg_attention1(self.avg_pool(negfusion_10))
        neg_max_at = self.neg_attention2(self.max_pool(negfusion_10))
        neg_at = self.sigmoid(neg_avg_at + neg_max_at)

        out = self.sigmoid(pos_at-neg_at)
        return out

class FUSION(nn.Module):
    def __init__(self,num_streak, num_mask):
        super(FUSION, self).__init__()
        self.num_streak = num_streak
        self.num_mask = num_mask

        fusion_list = []
        for i in range(num_streak):
            fusion_list.append(FUSION_BLOCK(num_mask))
        self.fusion_list = nn.ModuleList(fusion_list)

    def forward(self,streaks,masks):
        streak_sum_r = streak_sum_g = streak_sum_b = 0
        for i in range(self.num_streak):
            steaks_r = streaks[i][:,0,:,:].unsqueeze(1)
            steaks_g = streaks[i][:,1,:,:].unsqueeze(1)
            steaks_b = streaks[i][:,2,:,:].unsqueeze(1)

            match = self.fusion_list[i](masks,streaks[i])
            fusion_r = steaks_r*masks*match
            fusion_g = steaks_g*masks*match
            fusion_b = steaks_b*masks*match

            fusion_r = torch.sum(fusion_r,dim=1,keepdim=True)
            fusion_g = torch.sum(fusion_g,dim=1,keepdim=True)
            fusion_b = torch.sum(fusion_b,dim=1,keepdim=True)

            streak_sum_r = streak_sum_r + fusion_r
            streak_sum_g = streak_sum_g + fusion_g
            streak_sum_b = streak_sum_b + fusion_b
        streak_sum = torch.cat([streak_sum_r,streak_sum_g,streak_sum_b],1)

        return streak_sum








