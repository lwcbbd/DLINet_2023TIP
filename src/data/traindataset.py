import os
from data import srdata

class TrainDataset(srdata.SRData):
    def __init__(self, args, train=True, benchmark=False):
        super(TrainDataset, self).__init__(
            args, train=train, benchmark=benchmark
        )
        self.datasetname = args.dataset

    def _scan(self):
        names_hr, names_lr = super(TrainDataset, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(TrainDataset, self)._set_filesystem(dir_data)

        if self.datasetname in ['rain200L', 'rain200H', 'rain1400', 'rain300mixed']:
            self.apath = os.path.join('../dataset',self.datasetname,'train') # train data path
            self.ext = ('.png', '.png')
            self.dir_hr = os.path.join(self.apath, 'norain')
            self.dir_lr = os.path.join(self.apath, 'rain')
        elif self.datasetname in ['rain13k']:
            self.apath = os.path.join('../dataset',self.datasetname,'train') # train data path
            self.ext = ('.jpg', '.jpg')
            self.dir_hr = os.path.join(self.apath, 'norain')
            self.dir_lr = os.path.join(self.apath, 'rain')
        elif self.datasetname in ['spadata']:
            self.apath = '../' + 'dataset/' + self.datasetname + '/train/'  # train data path
            self.ext = ('.png', '.png')
            self.dir_hr = './data/spadata_train_target.txt'
            self.dir_lr = './data/spadata_train_input.txt'
        else:
            print("dataset name error!!!")



