import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm


class baseline_sentiment_bert(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        if freeze:
            self.freeze()
        self.feature_dim = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, nclasses)
        )

    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = self.classifier(pooled_output)
        return logits

    def freeze(self):

        for param in self.bert.parameters():
            param.requires_grad = False


class xlnet_base(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()
        self.nclasses = nclasses
        self.xlnet = transformers.XLNetForSequenceClassification.from_pretrained(
            "xlnet-base-cased", num_labels=nclasses)
        if freeze:
            self.freeze()

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs[0]
        return logits

    def freeze(self):
        for param in self.xlnet.transformer.parameters():
            param.requires_grad = False


class xlnet_large(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()
        self.nclasses = nclasses
        self.xlnet = transformers.XLNetForSequenceClassification.from_pretrained(
            "xlnet-large-cased", num_labels=nclasses)
        if freeze:
            self.freeze()

    def forward(self, input_ids, attention_mask):
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs[0]
        return logits

    def freeze(self):
        for param in self.model.transformer.parameters():
            param.requires_grad = False


class bert_base(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()
        self.nclasses = nclasses
        self.config = transformers.BertConfig(
            vocab_size=50_000

        )
        self.bert = transformers.BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=nclasses)
        if freeze:
            self.freeze()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs[0]
        return logits

    def freeze(self):
        for param in self.bert.bert.parameters():
            param.requires_grad = False


class bert_large(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()
        self.nclasses = nclasses
        self.bert = transformers.BertForSequenceClassification.from_pretrained(
            "bert-large-uncased", num_labels=nclasses)
        if freeze:
            self.freeze()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs[0]
        return logits

    def freeze(self):
        for param in self.bert.bert.parameters():
            param.requires_grad = False


class bert_multi(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()
        self.nclasses = nclasses
        self.bert = transformers.BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-uncased", num_labels=nclasses)
        if freeze:
            self.freeze()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs[0]
        return logits

    def freeze(self):
        for param in self.bert.bert.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    dev = torch.device('cpu')
    net = xlnet_sentiment(5).to(dev)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    tbar = tqdm(range(100))
    for iter_id in tbar:
        inps = {
            'input_ids': torch.rand(8, 100).long().to(dev),
            'attention_mask': torch.rand(8, 100).long().to(dev),
        }
        lbls = torch.rand(8, 1).long().to(dev)

        outs = net(inps['input_ids'], inps['attention_mask'])
        print(type(outs))
        print(type(lbls))
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        tbar.set_description_str(f'{iter_id}: {loss.item()}')
