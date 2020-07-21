import torch
import torch.nn as nn
from datasets.shopee import shopee_raw
from models.sentiment import baseline_sentiment_bert
from tqdm import tqdm

if __name__ == "__main__":
    dev = torch.device('cpu')
    dataset = shopee_raw('/home/ken/shopee_ws/sentiment/shopee-contest6/data/')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=False)

    net = baseline_sentiment_bert(5).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    tbar = tqdm(range(10))
    for iter_id in tbar:
        # inps = {
        #     'input_ids': torch.rand(8, 100).to(torch.int64).to(dev),
        #     'attention_mask': torch.rand(8, 100).to(torch.int64).to(dev),
        #     'targets': torch.randint(low=0, high=5, size=(8, 1)).flatten().to(dev),
        # }
        inps = next(iter(dataloader))
        print(inps['input_ids'].shape)
        print(inps['targets'].shape)
        lbls = inps['targets']

        outs = net(inps['input_ids'], inps['attention_mask'])
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        tbar.set_description_str(f'{iter_id}: {loss.item()}')
