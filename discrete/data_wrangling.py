import pandas as pd
from tqdm import tqdm

df = pd.read_csv(r'D:\GitHub\bias_free_modeling\data/discrete/goemotions.csv')

# voted
df_unique = df.drop_duplicates(subset=['text'])

labels = []
for text in tqdm(df_unique['text']):
    temp = df[df['text'] == text]
    most_voted_label = temp['label'].value_counts().idxmax()
    # check if there are multiple labels with the same number of votes
    if len(temp['label'].value_counts()) > 1:
        if temp['label'].value_counts().iloc[0] == temp['label'].value_counts().iloc[1]:
            most_voted_label = 'ambiguous'
    labels.append(most_voted_label)

df_unique['most_voted'] = labels

# save
df_unique.to_csv('D:\GitHub\bias_free_modeling\data/discrete/goemotions_with_ambiguous.csv', index=False)

# drop ambiguous
df_nonambiguous = df_unique[df_unique['most_voted'] != 'ambiguous']
# save
df_nonambiguous.to_csv(r'D:\GitHub\bias_free_modeling\data/discrete/goemotions_without_ambiguous.csv', index=False)

import pandas as pd

# load
df = pd.read_csv(r'D:\GitHub\bias_free_modeling\data/discrete/goemotions_without_ambiguous.csv')


emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

df = df[df['most_voted'].isin(emotions)]

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.4, random_state=42)
test, val = train_test_split(test, test_size=0.5, random_state=42)

# save
train.to_csv('data/discrete/train_hierarchical.csv', index=False)
test.to_csv('data/discrete/test_hierarchical.csv', index=False)
val.to_csv('data/discrete/val_hierarchical.csv', index=False)
