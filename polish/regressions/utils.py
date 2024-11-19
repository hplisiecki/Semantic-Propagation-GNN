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

batch = 400

dropout = 0.0
metric_names = ['norm_Valence_M', 'norm_Arousal_M']

with open(r'D:\GitHub\bias_free_modeling\data/polish/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)


with open(r'data/polish/parser_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

num_node_features = 3 # POS tag, Negation, Valence, Arousal, Dominance
model = SPropGNN(num_node_features, len(pos_tags) + 1 , len(parser_tag_categories) + 2, metric_names, dropout_prob=dropout)

# load
SAVE_DIR = r'D:\GitHub\bias_free_modeling\models\polish_golden'

model.load_state_dict(torch.load(SAVE_DIR))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.cuda()

import pandas as pd
def predict_text(text, batch_size = 100):
    # if text of type str, convert to list
    if isinstance(text, str):
        text = [text]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_text = pd.DataFrame({'text': text, 'norm_Valence_M' : [0 for _ in range(len(text))], 'norm_Arousal_M' : [0 for _ in range(len(text))], 'norm_Dominance_M' : [0 for _ in range(len(text))]})
    text_dataset = SPropDataset(df_text)
    text_dataloader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    predictions = {metric: [] for metric in metric_names}
    with torch.no_grad():
        for data in text_dataloader:
            data = data.to(device)
            outputs = model(data)
            for metric in metric_names:
                predictions[metric].extend([item.cpu().item() for item in outputs[metric]])

    return predictions
