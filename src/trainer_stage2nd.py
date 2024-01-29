import os
import math
from decimal import Decimal
import utility
import IPython
import torch
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage

def pretrain_load(args, model_stage1st, model_stage2nd, ckp_stage1st):
    if args.test_only:
        return None, model_stage2nd
    else:
        pretrain_dir = os.path.join(ckp_stage1st.dir, 'model/model_best.pt')
        model_stage1st.load(ckp_stage1st.dir,pre_train=pretrain_dir,resume=args.resume,cpu=args.cpu,mode='second_stage',args=args)
        model_stage2nd.load(ckp_stage1st.dir,pre_train=pretrain_dir,resume=args.resume,cpu=args.cpu,mode='second_stage',strict=False,args=args)

        return model_stage1st, model_stage2nd

class Trainer():
    def __init__(self, args, loader, model_stage1st, model_stage2nd, loss, ckp, ckp_stage1st):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.ckp_stage1st = ckp_stage1st
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model_stage1st, self.model = pretrain_load(self.args, model_stage1st, model_stage2nd, self.ckp_stage1st)
        self.loss = loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        if self.args.load != '.':
            print(ckp.dir)
            assert os.path.exists(ckp.dir + 'optimizer.pt')
            print('==============', ckp.dir + 'optimizer.pt')
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        # print('======>trian')
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        self.model_stage1st.eval()
        self.mask_loss = torch.nn.BCELoss()

        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.model.zero_grad()
            self.optimizer.zero_grad()

            mask_gt = self.model_stage1st.model.rain_detect(lr,hr)
            out, mask = self.model([lr], idx_scale)
            derainloss = self.loss(out, lr - hr)
            maskloss = self.mask_loss(mask, mask_gt.detach())
            loss = derainloss  + 10000*maskloss

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                ttt = 0
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()

        self.scheduler.step()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        # print('=========eval')
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                print(enumerate(tqdm_test))
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    out,_ = self.model([lr], idx_scale)

                    sr = lr - out
                    sr = utility.quantize(sr, self.args.rgb_range)  # restored background at the last stage
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=True
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.dataset,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else self.args.GPU_id)

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            img_cut_x = tensor.shape[2] % 8
            img_cut_y = tensor.shape[3] % 8
            tensor = tensor[:, :, 0:tensor.shape[2] - img_cut_x, 0:tensor.shape[3] - img_cut_y]
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
