import torch
import random
import os
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer_stage1st import Trainer as T_trainer
from trainer_stage2nd import Trainer as S_trainer
import multiprocessing
import time

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    all_param = num_params

    print('Total number of parameters: %d' % all_param)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_seed(1334)
    checkpoint = utility.checkpoint(args)

    print(args.mode)
    if checkpoint.ok:
        loader = data.Data(args)

        if args.mode == 'first_stage':
            model = model.Model(args, checkpoint, 'first_stage')
            loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = T_trainer(args, loader, model, loss, checkpoint)

        elif args.mode == 'second_stage':

            model_stage1st = model.Model(args, checkpoint, 'first_stage')
            model = model.Model(args, checkpoint, 'second_stage')
            if not args.test_only:
                loss = loss.Loss(args, checkpoint)
                checkpoint_stage1st = utility.pretrain_checkpoint(args)
            else:
                loss = None
                model_stage1st = None
                checkpoint_stage1st = None

            t = S_trainer(args, loader, model_stage1st, model, loss, checkpoint, checkpoint_stage1st)

        print_network(model)

        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()
    



