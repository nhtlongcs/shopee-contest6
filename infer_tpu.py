import os
import yaml
import torch
import pprint
import logging
import argparse
import warnings
import torch.nn as nn
import pytorch_lightning as pl
import torch_xla.core.xla_model as xm
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import *
from dataloaders import *
from torchsummary import summary
from utils.getter import get_instance
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(level=logging.ERROR)


class pipeline(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = get_instance(self.config['model'])
        self.loss = get_instance(self.config['loss'])

    def prepare_data(self):
        self.train_dataset = get_instance(self.config['dataset']['train'])
        self.val_dataset = get_instance(self.config['dataset']['val'])

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = self.loss(logits, batch['targets']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        loss = self.loss(logits, batch['targets'])
        acc = (logits.argmax(-1) == batch['targets']).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.mean(torch.stack([o['loss'] for o in outputs], dim=0))
        acc = torch.mean(torch.stack([o['acc'] for o in outputs], dim=0))
        out = {'val_loss': loss, 'val_acc': acc}
        return {**out, 'log': out}

    def train_dataloader(self):

        train_dataloader = get_instance(self.config['dataset']['train']['loader'],
                                        dataset=self.train_dataset, num_workers=4)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = get_instance(self.config['dataset']['val']['loader'],
                                      dataset=self.val_dataset, num_workers=4)
        return val_dataloader

    def configure_optimizers(self):
        optimizer = get_instance(self.config['optimizer'],
                                 params=self.model.parameters())
        return optimizer


def predict(config):

    device = xm.xla_device()
    config['dataset']['train']['args']['data_root_dir'] = './data/clean/'
    config['dataset']['train']['args']['infer'] = True
    test_dataset = get_instance(config['dataset']['train'])
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, drop_last=False)
    model = pipeline(config)
    checkpoint_path = config['trainer']['cp_dir']
    print('Loading model ...')
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model.eval()
    model.freeze()
    print('generating submission ... ')
    res = torch.Tensor().long().cpu()
    for x in tqdm(test_dataloader):
        inps = x['input_ids'].to(device)
        mask = x['attention_mask'].to(device)
        tmp = model(inps, mask)

        tmp = tmp.argmax(-1).cpu()
        res = torch.cat((res, tmp), 0)
    length = res.shape[0]
    if config['model']['args']['nclasses'] == 3:
        res = res + torch.Tensor([3]*length).long()
    else:
        res = res + torch.Tensor([1]*length).long()

    res_csv = res.tolist()
    idx = [x+1 for x in range(length)]
    df = pd.DataFrame(zip(idx, res_csv), columns=['review_id', 'rating'])
    print(length)
    df.to_csv(config['out'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='configs/train/tpu_colab_xlnet_full_unfreeze.yaml')
    parser.add_argument('--gpus', default=0)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--cp_dir', default='./cp')
    parser.add_argument('--out', default='./submit.csv')

    args = parser.parse_args()
    seed_everything(seed=args.seed)
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus
    config['debug'] = args.debug
    config['out'] = args.out
    config['trainer']['cp_dir'] = args.cp_dir
    predict(config)
