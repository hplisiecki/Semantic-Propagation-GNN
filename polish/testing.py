from polish.model import SPropGNN
from polish.dataset import SPropDataset
from polish.utils import load_data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from polish.training_loop import train_model
import os
import pickle
import wandb

# import pearson
from scipy.stats import pearsonr

train, test, val = load_data()

test_dataset = SPropDataset(test)

# save
torch.save(test_dataset, r'data/polish/test_hierarchical.pt')

# load
test_dataset = torch.load(r'data/polish/test_hierarchical.pt')

test_dataloader = DataLoader(test_dataset, batch_size=400, shuffle=False, drop_last=True)
predicted_metrics = ['norm_Valence_M', 'norm_Arousal_M']
metric_names = ['norm_Valence_M', 'norm_Arousal_M']
batch = 400

dropout = 0.0

with open(r'D:\GitHub\bias_free_modeling\data/polish/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)


with open(r'data/polish/parser_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

num_node_features = 3 # POS tag, Negation, Valence, Arousal, Dominance
model = SPropGNN(num_node_features, len(pos_tags) + 1 , len(parser_tag_categories) + 2, metric_names, dropout_prob=dropout)

# load
SAVE_DIR = r'D:\GitHub\bias_free_modeling\models\polish_golden'

model.load_state_dict(torch.load(SAVE_DIR))

criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.cuda()
criterion = criterion.cuda()


model.eval()
total_val_loss = 0
all_preds = {metric: [] for metric in metric_names}
all_labels = {metric: [] for metric in metric_names}
with torch.no_grad():
    for data in test_dataloader:
        data = data.to(device)
        outputs = model(data)
        preds = torch.cat([outputs[metric] for metric in metric_names], dim=1)
        valence = data.v.unsqueeze(1)
        arousal = data.a.unsqueeze(1)
        labels = torch.cat([valence, arousal], dim=1)
        loss = criterion(preds, labels)

        total_val_loss += loss.item() * data.num_graphs

        # Collect predictions and labels for correlation calculation
        for idx, metric in enumerate(metric_names):
            all_preds[metric].extend(outputs[metric].squeeze().tolist())
            all_labels[metric].extend(labels[:, idx].tolist())

avg_val_loss = total_val_loss / len(test_dataloader.dataset)
# Calculate correlation coefficients
epoch_correlations = {}
for metric in metric_names:
    pred_values = all_preds[metric]
    true_values = all_labels[metric]
    if len(set(pred_values)) > 1 and len(set(true_values)) > 1:
        corr, _ = pearsonr(pred_values, true_values)
    else:
        corr = 0.0  # Undefined correlation
    epoch_correlations[metric] = corr

# print the results
print(f"Valence correlation: {epoch_correlations['norm_Valence_M']}")
print(f"Arousal correlation: {epoch_correlations['norm_Arousal_M']}")
# and loss
print(f"Validation loss: {avg_val_loss}")

import pandas as pd
def predict_single_text(text):
    df_text = pd.DataFrame({'text': [text], 'norm_Valence_M' : [0.5], 'norm_Arousal_M' : [0.5], 'norm_Dominance_M' : [0.5]})
    text_dataset = SPropDataset(df_text)
    text_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False, drop_last=True)
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in text_dataloader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.cat([outputs[metric] for metric in predicted_metrics], dim=1)
            valence = data.v.unsqueeze(1)
            arousal = data.a.unsqueeze(1)
            labels = torch.cat([valence, arousal], dim=1)
            loss = criterion(preds, labels)

            total_val_loss += loss.item() * data.num_graphs

    return outputs

import spacy
nlp = spacy.load("pl_core_news_sm")

text= 'Nie za bardzo lubie matematykę'

doc = nlp(text)

edge_index = []
for token in doc:
    head = token.head.i
    index = token.i
    if head != index:
        # Add forward edge
        edge_index.append([head, index])


# final
# Valence correlation: 0.7153691107587314
# Arousal correlation: 0.6202863812986906
# Validation loss: 0.0254195436835289


# roberta model
# Valence_M: correlation - 0.876483573454043, pred_mean - 0.41013388633728026, real_mean - 0.41182500000000005, pred_sd - 0.23043739795684814, real_sd - 0.265400959634663
# Arousal_M: correlation - 0.7535152618312501, pred_mean - 0.3880662113428116, real_mean - 0.385175, pred_sd - 0.15754365921020508, real_sd - 0.2078406105047808