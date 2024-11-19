from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from tqdm import tqdm


# load val
val = pd.read_csv(r'data/english/val_dropped.csv')
analyzer = SentimentIntensityAnalyzer()


results = []
for text in tqdm(val['text']):
    vs = analyzer.polarity_scores(text)
    results.append(vs['compound'])

# correlate
print(f"Correlation between compound and norm_Valence_M: {np.corrcoef(val['norm_Valence_M'], results)[0, 1]}")


# Correlation between compound and norm_Valence_M: 0.4577709634968419
0