from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.traindataset')
            trainset = getattr(module_train, 'TrainDataset')(args)
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu
            )

        module_test = import_module('data.testdataset')
        testset = getattr(module_test, 'TestDataset')(args, train=False)
        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu
        )

