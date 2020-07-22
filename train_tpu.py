import os
import yaml
import torch
import pprint
import logging
import argparse
import warnings
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils import data
from torch.utils.data import DataLoader

from pytorch_lightning.trainer import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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


def train(config):
    sentiment_pipeline = pipeline(config)
    cp_dir = config['trainer']['cp_dir']
    cp_dir = os.path.join(
        cp_dir, config['model']['name']) + str(config.get('id', 'None'))
    checkpoint_callback = ModelCheckpoint(
        filepath=cp_dir + '/{epoch}-{train_loss:.3f}-{val_acc:.2f}')

    trainer = pl.Trainer(
        max_epochs=config['trainer']['nepochs'],
        num_tpu_cores=8,
        gpus=(1 if torch.cuda.is_available() else 0),
        check_val_every_n_epoch=config['trainer']['val_step'],
        checkpoint_callback=checkpoint_callback,
        default_root_dir='runs',
        logger=pl.loggers.TensorBoardLogger(
            'runs/', name=config['model']['name'], version=config.get('id', 'None'))
    )
    trainer.fit(sentiment_pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='configs/train/baseline_local_nhtlong_lightning.yaml')
    parser.add_argument('--gpus', default=0)
    parser.add_argument('--seed', default=123)
    parser.add_argument('--debug', default=False)

    args = parser.parse_args()
    seed_everything(seed=args.seed)
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config['gpus'] = args.gpus
    config['debug'] = args.debug
    train(config)
