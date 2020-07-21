import transformers


def tokenizer(pretrain=None):
    if pretrain == None:
        return transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    return transformers.BertTokenizer
