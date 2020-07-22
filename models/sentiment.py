import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm


class baseline_sentiment_bert(nn.Module):
    """Baseline model"""

    def __init__(self, nclasses, freeze=False):
        super().__init__()
        if freeze:
            self.freeze()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
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


if __name__ == "__main__":
    dev = torch.device('cpu')
    net = baseline_sentiment_bert(5).to(dev)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    tbar = tqdm(range(100))
    for iter_id in tbar:
        inps = {
            'input_ids': torch.rand(8, 100).to(dev),
            'attention_mask': torch.rand(8, 100).to(dev),
            'targets': torch.randint(low=0, high=2, size=(8)).to(dev)
        }
        lbls = torch.randint(low=0, high=2, size=(8)).to(dev),

        outs = net(inps)

        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        tbar.set_description_str(f'{iter_id}: {loss.item()}')
