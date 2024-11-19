import pandas as pd
from tqdm import tqdm

df = pd.read_csv(r'D:\GitHub\bias_free_modeling\data\english\emobank.csv')

# drop nans and duplicates from text
df = df.dropna(subset=['text'])
df = df.drop_duplicates(subset=['text'])


# normalize the metrics to 0 to 1
df['norm_Valence_M'] = (df['V'] - 1) / (5 - 1)
df['norm_Arousal_M'] = (df['A'] - 1) / (5 - 1)
df['norm_Dominance_M'] = (df['D'] - 1) / (5 - 1)


train = df[df['split'] == 'train']
test = df[df['split'] == 'test']
val = df[df['split'] == 'dev']

# save
train.to_csv(r'D:\GitHub\bias_free_modeling\data\english\train.csv', index=False)
test.to_csv(r'D:\GitHub\bias_free_modeling\data\english\test.csv', index=False)
val.to_csv(r'D:\GitHub\bias_free_modeling\data\english\val.csv', index=False)