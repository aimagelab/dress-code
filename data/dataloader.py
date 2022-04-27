import torch


class DataLoader(object):
    def __init__(self, opt, dataset, dist_sampler=False):
        super(DataLoader, self).__init__()
        if dist_sampler:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=opt.world_size, rank=opt.rank, shuffle=True)
        else:
            if opt.shuffle:
                train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
            else:
                train_sampler = None

        self.sampler = train_sampler
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
