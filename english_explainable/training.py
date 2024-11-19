from english_explainable.model import SPropGNN
from english.dataset import SPropDataset
from english.utils import load_data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from english.training_loop import train_model
import os
import pickle
import wandb


train, test, val = load_data()

redo = True
if os.path.exists(r'D:\GitHub\bias_free_modeling\data/english/train_hierarchical.pt') and redo == False:
    train_dataset = torch.load(r'D:\GitHub\bias_free_modeling\data/english/train_hierarchical.pt')
    val_dataset = torch.load(r'D:\GitHub\bias_free_modeling\data/english/val_hierarchical.pt')
else:
    train_dataset, val_dataset = SPropDataset(train), SPropDataset(val)
    # save datasets
    torch.save(train_dataset, r'D:\GitHub\bias_free_modeling\data/english/train_hierarchical.pt')
    torch.save(val_dataset, r'D:\GitHub\bias_free_modeling\data/english/val_hierarchical.pt')

batch = 400
metric_names = ['norm_Valence_M', 'norm_Arousal_M']

train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

with open(r'D:\GitHub\bias_free_modeling\data/english/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)

with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tag_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

dropout = 0.6
learning_rate = 0.00005
weight_decay = 0.4
num_epochs = 200


num_node_features = 3 # POS tag, Negation, Valence, Arousal, Dominance
model = SPropGNN(num_node_features, len(pos_tags) + 1, len(parser_tag_categories) + 2, metric_names,
                     dropout_prob=dropout)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
SAVE_DIR = r'D:\GitHub\bias_free_modeling\models\scalar_scaling_english_dimensions'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = model.to(device)
criterion = criterion.to(device)


train_model(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, metric_names, device,
            save_dir = SAVE_DIR, use_wandb = False, scheduler = None)
