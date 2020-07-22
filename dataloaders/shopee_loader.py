import torch


def shopee_loader(dataset, batch_size, shuffle=False, num_workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)
