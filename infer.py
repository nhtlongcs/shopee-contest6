import csv
import os
from tqdm import tqdm

import pandas
import torch
import yaml
from PIL import Image
from torchvision import transforms as tvtf

from utils.getter import get_instance

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
model.eval()
with torch.no_grad():
    data = list(pd.read_csv(csv_test_dir)['review'])
    result = []
    cnt = 0
    for item in tqdm(data):

        review = str(item).lower() + "[SEP] [CLS]"
        encoding = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        review = encoding['input_ids'].flatten().unsqueeze(0).to(device)
        mask = encoding['attention_mask'].flatten().unsqueeze(0).to(device)

        outputs = model(review, mask)
        probs = torch.softmax(outputs, dim=1).to('cpu')
        pred = probs.argmax(dim=1).numpy()
        result.append(pred)

result = list(map(int, result))

df = pandas.DataFrame(data={"id": data, "rating": result})
df.to_csv("./submission.csv", sep=',', index=False)
