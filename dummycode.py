import torch
import torch.nn as nn
from datasets.shopee import shopee_raw, shopee_dummy
from models.sentiment import baseline_sentiment_bert
from tqdm import tqdm

if __name__ == "__main__":
    dev = torch.device('cpu')
    dataset = shopee_dummy('/home/ken/shopee_ws/sentiment/shopee-contest6/data/clean/full/')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=False)

    net = baseline_sentiment_bert(5).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    tbar = tqdm(range(10))
    for iter_id in tbar:
        inps = next(iter(dataloader))
        print(inps['input_ids'].shape)
        print(inps['targets'].shape)
        lbls = inps['targets']

        outs = net(inps['input_ids'], inps['attention_mask'])
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        tbar.set_description_str(f'{iter_id}: {loss.item()}')
