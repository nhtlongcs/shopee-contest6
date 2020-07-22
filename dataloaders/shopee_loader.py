import torch


def shopee_loader(dataset, batch_size, shuffle=False, num_workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
