from discrete.model import SPropGNN
from discrete.dataset import SPropDataset
from discrete.utils import load_data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import pickle
# pearsonr
# import F
import torch.nn.functional as F

train, test, val = load_data()

test_dataset = SPropDataset(test)

test_dataloader = DataLoader(test_dataset, batch_size=400, shuffle=False)
batch = 400

num_epochs = 400
learning_rate = 0.005
dropout = 0.4
weight_decay = 0.6

with open(r'D:\GitHub\bias_free_modeling\data/english/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)

with open(r'data/english/parser_tag_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

with open(r'data/discrete/label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

num_node_features = 9 # emotion and sentence location
model = SPropGNN(num_node_features, len(pos_tags) + 1 , len(parser_tag_categories) + 2, len(label_map), dropout_prob=dropout)

SAVE_DIR = r'D:\GitHub\bias_free_modeling\models\discrete_golden'
# load model
model.load_state_dict(torch.load(SAVE_DIR))
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)


model.eval()
total_val_loss = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for data in test_dataloader:
        data = data.to(device)
        outputs = model(data)
        labels = data.y
        loss = criterion(outputs, labels)

        total_val_loss += loss.item() * data.num_graphs

        # Apply softmax to outputs to get probabilities, then use argmax to get predicted classes
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Collect predictions and labels for evaluation
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np

# Flatten all_preds and all_labels to single arrays after collecting data
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate metrics per emotion
print("Performance Metrics per Emotion:")
for emotion, idx in label_map.items():
    # Extract predictions and labels for the current emotion
    emotion_labels = (all_labels == idx).astype(int)
    emotion_preds = (all_preds == idx).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(emotion_labels, emotion_preds)
    precision = precision_score(emotion_labels, emotion_preds, zero_division=0)
    recall = recall_score(emotion_labels, emotion_preds, zero_division=0)
    f1 = f1_score(emotion_labels, emotion_preds, zero_division=0)

    print(f"Emotion: {emotion}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("-" * 30)

# Calculate overall metrics
overall_accuracy = accuracy_score(all_labels, all_preds)
overall_precision = precision_score(all_labels, all_preds, average='weighted')
overall_recall = recall_score(all_labels, all_preds, average='weighted')
overall_f1 = f1_score(all_labels, all_preds, average='weighted')

print("\nOverall Metrics")
print(f"  Overall Accuracy: {overall_accuracy:.4f}")
print(f"  Overall Precision (Weighted): {overall_precision:.4f}")
print(f"  Overall Recall (Weighted): {overall_recall:.4f}")
print(f"  Overall F1 Score (Weighted): {overall_f1:.4f}")


# Performance Metrics per Emotion:
# Emotion: anger
#   Accuracy: 0.7956
#   Precision: 0.6593
#   Recall: 0.8226
#   F1 Score: 0.7320
# ------------------------------
# Emotion: disgust
#   Accuracy: 0.8890
#   Precision: 0.5067
#   Recall: 0.3519
#   F1 Score: 0.4153
# ------------------------------
# Emotion: fear
#   Accuracy: 0.9315
#   Precision: 0.7209
#   Recall: 0.5962
#   F1 Score: 0.6526
# ------------------------------
# Emotion: joy
#   Accuracy: 0.9222
#   Precision: 0.7636
#   Recall: 0.7778
#   F1 Score: 0.7706
# ------------------------------
# Emotion: sadness
#   Accuracy: 0.8890
#   Precision: 0.6154
#   Recall: 0.4885
#   F1 Score: 0.5447
# ------------------------------
# Emotion: surprise
#   Accuracy: 0.9066
#   Precision: 0.6667
#   Recall: 0.6364
#   F1 Score: 0.6512
# ------------------------------
# Overall Metrics
#   Overall Accuracy: 0.6670
#   Overall Precision (Weighted): 0.6614
#   Overall Recall (Weighted): 0.6670
#   Overall F1 Score (Weighted): 0.6579



text = 'I am not happy.'
import pandas as pd

df_text = pd.DataFrame({'text': [text], 'most_voted': ['anger']})
text_dataset = SPropDataset(df_text)
text_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False, drop_last=True)
model.eval()
total_val_loss = 0
with torch.no_grad():
    for data in text_dataloader:
        data = data.to(device)
        outputs = model(data)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        emotion = list(label_map.keys())[list(label_map.values()).index(preds.item())]
# Access activations
activations = model.activations

# Visualize input features
# features = visualize_input_features(data)
from shed.explainable_polish.explaining import visualize_scaling_factors

# Visualize scaling factors for the first GNN layer
visualize_scaling_factors(data, activations, layer_index=0)