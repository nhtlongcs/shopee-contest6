import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch import Tensor
from sklearn.model_selection import train_test_split
import transformers


class shopee_raw(data.Dataset):
    def __init__(self, data_root_dir, max_len=100, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.root_dir = data_root_dir

        csv_train_dir = self.root_dir + 'train.csv'
        csv_val_dir = self.root_dir + 'val.csv'

        if self.is_train:
            self.reviews = list(pd.read_csv(csv_train_dir)['review'])
            self.targets = list(pd.read_csv(csv_train_dir)['rating'])
        elif self.is_train == False:
            self.reviews = list(pd.read_csv(csv_val_dir)['review'])
            self.targets = list(pd.read_csv(csv_val_dir)['rating'])

        self.targets = list(map(int, self.targets))
        self.tokenizer = self.get_tokenizer('bert-base-uncased')
        self.max_len = max_len

    def __getitem__(self, idx):

        review = str(self.reviews[idx])
        target = self.targets[idx]-1  # 1-> 5 map to 0 -> 4
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # return self.tokenizer.tokenize(review), target
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target).long()
        }

    def __len__(self):
        return len(self.reviews)

    def get_tokenizer(self, pretrain=None):
        if pretrain == None:
            return transformers.BertTokenizer
        return transformers.BertTokenizer.from_pretrained(pretrain)


class shopee_xlnet(data.Dataset):
    def __init__(self, data_root_dir, max_len=100, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.root_dir = data_root_dir

        csv_train_dir = self.root_dir + 'train.csv'
        csv_val_dir = self.root_dir + 'val.csv'

        if self.is_train:
            self.reviews = list(pd.read_csv(csv_train_dir)['review'])
            self.targets = list(pd.read_csv(csv_train_dir)['rating'])
        elif self.is_train == False:
            self.reviews = list(pd.read_csv(csv_val_dir)['review'])
            self.targets = list(pd.read_csv(csv_val_dir)['rating'])

        self.targets = list(map(int, self.targets))
        self.tokenizer = self.get_tokenizer('bert-base-uncased')
        self.max_len = max_len

    def __getitem__(self, idx):

        review = str(self.reviews[idx])
        target = self.targets[idx]-1  # 1-> 5 map to 0 -> 4
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # return self.tokenizer.tokenize(review), target
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target).long()
        }

    def __len__(self):
        return len(self.reviews)

    def get_tokenizer(self, pretrain=None):
        if pretrain == None:
            return transformers.XLNetTokenizer
        return transformers.XLNetTokenizer.from_pretrained('xlnet-large-cased')


# test
if __name__ == "__main__":
    dataset = shopee_xlnet(
        '/home/ken/shopee_ws/sentiment/shopee-contest6/data/clean/full/')
    print('++++++++++++++++++++++++++++')
    print(dataset[2]['review_text'])
    print('++++++++++++++++++++++++++++')
    print(dataset[2]['input_ids'])
    print('++++++++++++++++++++++++++++')
    print(dataset[2]['attention_mask'])
    print('++++++++++++++++++++++++++++')
    print(dataset[2]['targets'])
