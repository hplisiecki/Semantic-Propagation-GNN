import pandas as pd

def load_data():
    train = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\polish\train_set_prepared.csv')
    test = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\polish\test_set_prepared.csv')
    val = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\polish\val_set_prepared.csv')

    return train, test, val