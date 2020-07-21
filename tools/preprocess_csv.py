import pandas as pd
outpath = '/home/ken/shopee_ws/sentiment/shopee-contest6/data/'
train = pd.read_csv(
    '/home/ken/shopee_ws/sentiment/dataraw/val.csv')
train = train[['review', 'rating']]
train.to_csv(outpath + 'val.csv', index=False)
