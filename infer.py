import csv
import os
from tqdm import tqdm

import pandas
import torch
import yaml
from PIL import Image

from utils.getter import get_instance

if __name__ == "__main__":

    # config_path = '/content/shopee-contest2/configs/train/baseline_colab.yaml'
    cp_model_dir = './cp/bert_multi3/best_metric_Accuracy.pth'
    csv_test_dir = './data/clean/test.csv'

    config = torch.load(cp_model_dir).device('cpu')['config']

    dev_id = 'cuda:{}'.format(config['gpus']) \
        if torch.cuda.is_available() and config.get('gpus', None) is not None \
        else 'cpu'
    device = torch.device(dev_id)
    model = get_instance(config['model']).to(device)
    model.load_state_dict(torch.load(cp_model_dir)['model_state_dict'])

    print('load weights-----------------------')

    # Classify
    print('generate submission----------------')

    dataset = get_instance(config['dataset']['infer'])
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=False)

    # tbar.set_description_str(f'{iter_id}: {loss.item()}')

    model.eval()
    with torch.no_grad():
    for inps in tqdm(dataloader):
        # inps = next(iter(dataloader))
        print(inps['review_text'])
        print("===========")
        outs = model(inps['input_ids'], inps['attention_mask'])
        # tbar.set_description_str(f'{iter_id}: {loss.item()}')
        probs = torch.softmax(outputs, dim=1).to('cpu')
        pred = probs.argmax(dim=1).numpy()
        result.append(pred)

    result = list(map(int, result))
    df = pandas.DataFrame(
        data={"id": [x + 1 for x in range(len(result))], "rating": result})
    df.to_csv("./submission.csv", sep=',', index=False)
