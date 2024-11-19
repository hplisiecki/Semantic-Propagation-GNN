import pandas as pd


def load_data():
    train = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\english\train.csv')
    test = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\english\test.csv')
    val = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\english\val.csv')
    return train, test, val