import pandas as pd


def load_data():
    train = pd.read_csv(r'D:\GitHub\bias_free_modeling\data/discrete/train_hierarchical.csv')
    test = pd.read_csv(r'D:\GitHub\bias_free_modeling\data/discrete/test_hierarchical.csv')
    val = pd.read_csv(r'D:\GitHub\bias_free_modeling\data/discrete/val_hierarchical.csv')

    return train, test, val