import pandas as pd
import spacy
from english.word_emo_prediction import get_valence_arousal_dominance
import pickle
import requests
import numpy as np
from tqdm import tqdm
from english.utils import load_data


# load test
train, test, val = load_data()


nlp = spacy.load('en_core_web_sm')

stopwords_url = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
stop_words = requests.get(stopwords_url).text.splitlines()


results = {'valence': [], 'arousal': [], 'dominance': []}
words = set()
for text in tqdm(test['text']):
    doc = nlp(text)
    for token in doc:
        if (not token.is_punct and token.text.lower() not in stop_words):
            words.add(token.text)

valence, arousal, dominance = get_valence_arousal_dominance(list(words))
valence_dict = {word: v for word, v in zip(words, valence)}
arousal_dict = {word: a for word, a in zip(words, arousal)}
dominance_dict = {word: d for word, d in zip(words, dominance)}

for text in tqdm(test['text']):
    words = []
    doc = nlp(text)
    for token in doc:
        if (not token.is_punct and token.text.lower() not in stop_words):
            words.append(token.text)

    results['valence'].append(np.mean([valence_dict[word] for word in words]))
    results['arousal'].append(np.mean([arousal_dict[word] for word in words]))
    results['dominance'].append(np.mean([dominance_dict[word] for word in words]))

# to dataframe
results = pd.DataFrame(results)
results['text'] = test['text'].values
metric_columns = ['norm_Valence_M', 'norm_Arousal_M', 'norm_Dominance_M']
for metric in metric_columns:
    results[metric] = test[metric].values

# drop nans

# to csv
results.to_csv(r'data/english/baseline_lexicon.csv', index=False)

# dropna from valence, arousal, dominance
results = results.dropna(subset=['valence', 'arousal', 'dominance'])


# calculate correlations
metric_columns = ['norm_Valence_M', 'norm_Arousal_M', 'norm_Dominance_M']
for metric in metric_columns:
    print(f"Correlation between {metric} and valence: {np.corrcoef(results[metric], results['valence'].values)[0, 1]}")
    print(f"Correlation between {metric} and arousal: {np.corrcoef(results[metric], results['arousal'].values)[0, 1]}")
    print(f"Correlation between {metric} and dominance: {np.corrcoef(results[metric], results['dominance'].values)[0, 1]}")


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
# Correlation between norm_Valence_M and valence: 0.4462611608746312
# Correlation between norm_Valence_M and arousal: -0.17248452827285873
# Correlation between norm_Valence_M and dominance: 0.362142619622098
# Correlation between norm_Arousal_M and valence: -0.0568292245511334
# Correlation between norm_Arousal_M and arousal: 0.24633728117943882
# Correlation between norm_Arousal_M and dominance: -0.07761155107806965
# Correlation between norm_Dominance_M and valence: 0.1417790350552436
# Correlation between norm_Dominance_M and arousal: -0.03318823027550908
# Correlation between norm_Dominance_M and dominance: 0.1480117918465923