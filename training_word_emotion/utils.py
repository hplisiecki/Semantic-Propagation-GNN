import pandas as pd
import random
import numpy as np
import torch

def load_data(emotion):
    train = pd.read_csv(f'data/emolex/train_{emotion}.csv')
    test = pd.read_csv(f'data/emolex/test_{emotion}.csv')
    val = pd.read_csv(f'data/emolex/val_{emotion}.csv')
    return train, test, val


def check_max_token_length(tokenizer, emotion, df = None):
    if df is None:
        train = pd.read_csv(f'data/emolex/train_{emotion}.csv')
        test = pd.read_csv(f'data/emolex/test_{emotion}.csv')
        val = pd.read_csv(f'data/emolex/val_{emotion}.csv')
        texts = pd.concat([train.word, val.word, test.word], axis=0)
    else:
      texts = df['word']
    max_len = 0
    for text in texts:
        token_length = len(tokenizer.encode(text))
        if  token_length > max_len:
            max_len = token_length
    return max_len

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
