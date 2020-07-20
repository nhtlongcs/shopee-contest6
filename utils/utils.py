
import numpy as np
import torch
import torchvision
import os
from torch.nn import Module, Tensor
from typing import Optional, List, Tuple


def load_file_from_folder(folder_path):
    assert folder_path != None, "Folder Path is empty!"
    return os.listdir(folder_path)


def execute_filename(filename):
    filename = filename.split('-')[1]
    return filename


def rescale(img):
    return (img - torch.min(img))/(torch.max(img) - torch.min(img))


class NormMaxMin():
    def __call__(self, x):
        return (x.float() - torch.min(x)) / (torch.max(x) - torch.min(x))


def save_model(model: Module, path: str):
    torch.save(model.state_dict(), path)
    print("Model saved")


def load_model(model: Module, path: str):
    model.load_state_dict(torch.load(path))
    print("Model loaded")
