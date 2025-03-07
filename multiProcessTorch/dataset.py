import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

class MyDataset(Dataset):
    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return torch.randn(10), torch.randn(10)

def prepare_dataloader(rank, world_size, batch_size=32):
    dataset = MyDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # 是否打乱数据
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    return dataloader