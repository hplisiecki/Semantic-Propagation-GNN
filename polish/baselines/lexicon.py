import pandas as pd
import spacy
from polish.word_emo_prediction import get_valence_arousal
import pickle
import requests
import numpy as np
from tqdm import tqdm
from polish.utils import load_data


# load test
train, test, val = load_data()


nlp = spacy.load('pl_core_news_sm')

# read stopwords from text file
with open(r'D:\GitHub\bias_free_modeling\data\polish\stopwords.txt', 'r') as f:
    stop_words = f.read().splitlines()

results = {'valence': [], 'arousal': []}
words = set()
for text in tqdm(test['text']):
    doc = nlp(text)
    for token in doc:
        if (not token.is_punct and token.text.lower() not in stop_words):
            words.add(token.text)

valence, arousal = get_valence_arousal(list(words))
valence_dict = {word: v for word, v in zip(words, valence)}
arousal_dict = {word: a for word, a in zip(words, arousal)}

for text in tqdm(test['text']):
    words = []
    doc = nlp(text)
    for token in doc:
        if (not token.is_punct and token.text.lower() not in stop_words):
            words.append(token.text)

    results['valence'].append(np.mean([valence_dict[word] for word in words]))
    results['arousal'].append(np.mean([arousal_dict[word] for word in words]))

# to dataframe
results = pd.DataFrame(results)
results['text'] = test['text'].values
metric_columns = ['norm_Valence_M', 'norm_Arousal_M']
for metric in metric_columns:
    results[metric] = test[metric].values

# drop nans

# to csv
results.to_csv(r'data/polish/baseline_lexicon.csv', index=False)

# dropna from valence, arousal, dominance
results = results.dropna(subset=['valence', 'arousal'])


# calculate correlations
metric_columns = ['norm_Valence_M', 'norm_Arousal_M']
for metric in metric_columns:
    print(f"Correlation between {metric} and valence: {np.corrcoef(results[metric], results['valence'].values)[0, 1]}")
    print(f"Correlation between {metric} and arousal: {np.corrcoef(results[metric], results['arousal'].values)[0, 1]}")


# with stopwords
# Correlation between norm_Valence_M and valence: 0.4211113841898539
# Correlation between norm_Valence_M and arousal: -0.1660434115846341
# Correlation between norm_Valence_M and dominance: 0.3190804071619821
# Correlation between norm_Arousal_M and valence: -0.058906006359248336
# Correlation between norm_Arousal_M and arousal: 0.21899208210346838
# Correlation between norm_Arousal_M and dominance: -0.10200232281503063
# Correlation between norm_Dominance_M and valence: 0.11317522132094177
# Correlation between norm_Dominance_M and arousal: -0.019243307120714873
# Correlation between norm_Dominance_M and dominance: 0.1105045959875824


# without stopwords
# Correlation between norm_Valence_M and valence: 0.5735824775680709
# Correlation between norm_Valence_M and arousal: -0.2715564195348391
# Correlation between norm_Arousal_M and valence: -0.18681315483007144
# Correlation between norm_Arousal_M and arousal: 0.33161612087381387