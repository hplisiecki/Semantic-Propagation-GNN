"""
This file creates the label map for discrete emotions for the goemotion dataset.
"""

import pandas as pd
import pickle
from discrete.utils import load_data


# get the train, test, val data
train, test, val = load_data()

# concatenate
df = pd.concat([train, test, val])

# create the label map out of all unique labels
label_map = {label: idx for idx, label in enumerate(sorted(df['most_voted'].unique()))}

# save the label map
with open(r'data/discrete/label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)
