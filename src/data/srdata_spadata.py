import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.datasetname = args.dataset

        print('prepare data...')
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = self.apath+ 'bin/'
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        # print(list_hr)
        if args.ext.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr, list_lr = self._scan()
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
            self.images_lr = [
                self._check_and_load(args.ext, l, self._name_lrbin(s)) \
                for s, l in zip(self.scale, list_lr)
            ]
        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                for h in list_hr:
                    h = h.strip()
                    self.images_hr.append(h)
                for l in list_lr:
                    l = l.strip()
                    self.images_lr[0].append(l)
            elif args.ext.find('sep') >= 0:
                self.images_hr, self.images_lr = [], [[] for _ in self.scale]
                for h in list_hr:
                    h = h.strip()
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    b_name = os.path.basename(b)
                    b_path = b.replace(b_name,'')
                    os.makedirs(
                        b_path,
                        exist_ok=True
                    )
                    self.images_hr.append(b)
                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )
                for l in list_lr:
                    l = l.strip()
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    b_name = os.path.basename(b)
                    b_path = b.replace(b_name, '')
                    os.makedirs(
                        b_path,
                        exist_ok=True
                    )
                    self.images_lr[0].append(b)
                    self._check_and_load(
                        args.ext, [l], b,  verbose=True, load=False
                    )

        if train:
            self.repeat \
                = args.test_every / (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = open(self.dir_hr,'r').readlines()
        names_lr = open(self.dir_lr,'r').readlines()
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        # self.ext = ('.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR.pt'.format(self.split)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        # l.pop()
        # f = f.rstrip()
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f: ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f: pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self.get_patch(lr, hr)
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )

        return lr_tensor, hr_tensor, filename

    def __len__(self):
        if self.train:
            return int(len(self.images_hr) * self.repeat)
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
            lr = f_lr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
                lr = imageio.imread(f_lr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f: hr = np.load(_f, allow_pickle=True)[0]['image']
                with open(f_lr, 'rb') as _f: lr = np.load(_f, allow_pickle=True)[0]['image']

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
           # print('****preparte data****')
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
               # print('****use augment****')
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih, 0:iw]
            #hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
