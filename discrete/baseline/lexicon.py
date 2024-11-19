from discrete.utils import load_data
import pandas as pd
import spacy
from discrete.word_emo_prediction_discrete import get_discrete_emotions
import pickle
import requests
import numpy as np
from tqdm import tqdm

train, test, val = load_data()



nlp = spacy.load('en_core_web_sm')



# Define a set of punctuation marks to retain

stopwords_url = "https://gist.githubusercontent.com/sebleier/554280/raw/7e0e4a1ce04c2bb7bd41089c9821dbcf6d0c786c/NLTK's%2520list%2520of%2520english%2520stopwords"
stop_words = requests.get(stopwords_url).text.splitlines()
emotions = ['surprise', 'anger', 'sadness', 'joy', 'fear', 'disgust']

results = {emotion: [] for emotion in emotions}
words = set()
for text in tqdm(test['text']):
    doc = nlp(text)
    for token in doc:
        if (not token.is_punct and token.text.lower() not in stop_words):
            words.add(token.text)

emotions_dict = get_discrete_emotions(list(words))
# valence_dict = {word: v for word, v in zip(words, valence)}
# arousal_dict = {word: a for word, a in zip(words, arousal)}
# dominance_dict = {word: d for word, d in zip(words, dominance)}

for text in tqdm(test['text']):
    words = []
    doc = nlp(text)
    for token in doc:
        if (not token.is_punct and token.text.lower() not in stop_words):
            words.append(token.text)

    for emotion in emotions:
        results[emotion].append(np.mean([emotions_dict[emotion][word] for word in words]))

results = pd.DataFrame(results)
# save
results.to_csv(r'data/discrete/baseline_lexicon.csv', index=False)
# argmax
most_voted = results.idxmax(axis=1)
results['most_voted'] = most_voted
# save
results.to_csv(r'data/discrete/baseline_lexicon_most_voted.csv', index=False)

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np

# Flatten all_preds and all_labels to single arrays after collecting data
all_preds = results['most_voted'].values
all_labels = test['most_voted'].values

# Calculate metrics per emotion

# Define the emotion labels and map them to integers
emotions = ['surprise', 'anger', 'sadness', 'joy', 'fear', 'disgust']
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}

# Convert all_preds and all_labels from strings to numeric labels
all_preds = [emotion_to_idx[pred] for pred in results['most_voted'].values]
all_labels = [emotion_to_idx[label] for label in test['most_voted'].values]

# Calculate metrics per emotion
emotion_metrics = {}
for emotion in emotions:
    # Get the index for the current emotion
    emotion_idx = emotion_to_idx[emotion]

    # Calculate metrics for the current emotion (one-vs-rest approach)
    y_true = np.array(all_labels) == emotion_idx
    y_pred = np.array(all_preds) == emotion_idx

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    emotion_metrics[emotion] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Calculate overall metrics
overall_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
overall_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
overall_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
overall_accuracy = accuracy_score(all_labels, all_preds)

# Display metrics
print("Per-emotion metrics:")
for emotion, metrics in emotion_metrics.items():
    print(
        f"{emotion}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1-Score={metrics['f1_score']:.2f}")

print("\nOverall metrics:")
print(
    f"Precision={overall_precision:.2f}, Recall={overall_recall:.2f}, F1-Score={overall_f1:.2f}, Accuracy={overall_accuracy:.2f}")


# Per-emotion metrics:
# surprise: Precision=0.30, Recall=0.26, F1-Score=0.28
# anger: Precision=0.72, Recall=0.29, F1-Score=0.41
# sadness: Precision=0.31, Recall=0.06, F1-Score=0.10
# joy: Precision=0.29, Recall=0.91, F1-Score=0.44
# fear: Precision=0.61, Recall=0.22, F1-Score=0.32
# disgust: Precision=0.24, Recall=0.31, F1-Score=0.27
# Overall metrics:
# Precision=0.41, Recall=0.34, F1-Score=0.30, Accuracy=0.35
