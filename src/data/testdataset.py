import os
from data import srdata

class TestDataset(srdata.SRData):
    def __init__(self, args, train=True, benchmark=False):
        super(TestDataset, self).__init__(
            args, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(TestDataset, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(TestDataset, self)._set_filesystem(dir_data)

        if self.datasetname in ['rain200L', 'rain200H', 'rain1400', 'rain300mixed']:
            self.apath = os.path.join('../dataset',self.datasetname,'test') # train data path
            self.ext = ('.png', '.png')
            self.dir_hr = os.path.join(self.apath, 'norain')
            self.dir_lr = os.path.join(self.apath, 'rain')
        elif self.datasetname in ['rain13k']:
            subdatasetname = self.args.subdataset
            self.apath = os.path.join('../dataset', 'rain13k', 'test', subdatasetname,) # train data path
            self.ext = ('.png', '.png') if subdatasetname != 'Test2800' else ('.jpg', '.jpg')
            self.dir_hr = os.path.join(self.apath, 'target')
            self.dir_lr = os.path.join(self.apath, 'input')
        elif self.datasetname in ['spadata']:
            self.apath = '../' + 'dataset/' + self.datasetname + '/test/'  # train data path
            self.ext = ('.png', '.png')
            self.dir_hr = './data/spadata_test_target.txt'
            self.dir_lr = './data/spadata_test_input.txt'
        else:
            print("dataset name error!!!")



