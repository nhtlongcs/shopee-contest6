import pickle
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
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.root_dir = data_root_dir
        csv_train_dir = self.root_dir + 'train.csv'
        csv_val_dir = self.root_dir + 'val.csv'
        print(csv_train_dir)
        print(csv_val_dir)
        print(is_train)
        if self.is_train:
            self.reviews = list(pd.read_csv(csv_train_dir)['review'])[:585]
            self.targets = list(pd.read_csv(csv_train_dir)['rating'])[:585]
        elif self.is_train == False:
            self.reviews = list(pd.read_csv(csv_val_dir)['review'])[:585]
            self.targets = list(pd.read_csv(csv_val_dir)['rating'])[:585]

        self.targets = list(map(int, self.targets))
        self.tokenizer = self.get_tokenizer('bert-base-uncased')
        self.max_len = max_len
        # print(self.tokenizer.vocab)

    def __getitem__(self, idx):

        review = str(self.reviews[idx]).lower() + "[SEP] [CLS]"
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


class shopee_dummy(shopee_raw):
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super(shopee_dummy, self).__init__(
            data_root_dir, max_len=256, is_train=is_train)

    def get_tokenizer(self, pretrain=None):
        tokenizer = transformers.BertTokenizer.from_pretrained(
            './tokenizer30000/bert', max_len=256)
        return tokenizer


class shopee_xlnet_base(shopee_raw):
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super(shopee_xlnet_base, self).__init__(
            data_root_dir, max_len=256, is_train=is_train)

    def get_tokenizer(self, pretrain=None):
        return transformers.XLNetTokenizer.from_pretrained('xlnet-base-cased')


class shopee_xlnet_large(shopee_raw):
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super(shopee_xlnet_large, self).__init__(
            data_root_dir, max_len=256, is_train=is_train)

    def get_tokenizer(self, pretrain=None):
        if pretrain == None:
            return transformers.XLNetTokenizer
        return transformers.XLNetTokenizer.from_pretrained('xlnet-large-cased')


class shopee_bert_base(shopee_raw):
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super(shopee_bert_base, self).__init__(
            data_root_dir, max_len=256, is_train=is_train)

    def get_tokenizer(self, pretrain=None):
        return transformers.BertTokenizer.from_pretrained('./tokenizer30000/bert', max_len=256)


class shopee_bert_large(shopee_raw):
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super(shopee_bert_large, self).__init__(
            data_root_dir, max_len=256, is_train=is_train)

    def get_tokenizer(self, pretrain=None):
        return transformers.BertTokenizer.from_pretrained('./tokenizer30000/bert', max_len=256)


class shopee_bert_multi(shopee_raw):
    def __init__(self, data_root_dir, max_len=256, is_train=True):
        super(shopee_bert_multi, self).__init__(
            data_root_dir, max_len=256, is_train=is_train)

    def get_tokenizer(self, pretrain=None):
        return transformers.BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


# test
if __name__ == "__main__":
    dataset = shopee_dummy(
        '/home/ken/shopee_ws/sentiment/shopee-contest6/data/clean/full/')
    # print(dataset[2]['review_text'])
    # print(dataset[2]['input_ids'])
    # print(dataset[2]['attention_mask'])
    # print(dataset[2]['targets'])
