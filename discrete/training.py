from discrete.model import SPropGNN
from discrete.dataset import SPropDataset
from discrete.utils import load_data
from torch_geometric.loader import DataLoader
from general_utils import set_seed
import torch
import torch.nn as nn
import torch.optim as optim
from discrete.training_loop import train_model
import os
import pickle
import wandb

seed = 42
worker_init = set_seed(seed)

train, test, val = load_data()

redo = False
if os.path.exists(r'data/discrete/train_hierarchical.pt') and redo == False:
    train_dataset = torch.load(r'data/discrete/train_hierarchical.pt')
    val_dataset = torch.load(r'data/discrete/val_hierarchical.pt')
else:
    train_dataset, val_dataset = SPropDataset(train), SPropDataset(val)
    # save datasets
    torch.save(train_dataset, r'data/discrete/train_hierarchical.pt')
    torch.save(val_dataset, r'data/discrete/val_hierarchical.pt')

batch = 400

num_epochs = 200
learning_rate = 0.005
dropout = 0.4
weight_decay = 0.6

train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, worker_init_fn=worker_init)
val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=True, worker_init_fn=worker_init)

with open(r'D:\GitHub\bias_free_modeling\data/english/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)

with open(r'D:\GitHub\bias_free_modeling\data/english/parser_tag_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

with open(r'D:\GitHub\bias_free_modeling\data/discrete/label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

num_node_features = 9 # emotion and sentence location
model = SPropGNN(num_node_features, len(pos_tags) + 1 , len(parser_tag_categories) + 2, len(label_map), dropout_prob=dropout)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
SAVE_DIR = r'D:\GitHub\bias_free_modeling\models\discrete_golden'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = model.to(device)
criterion = criterion.to(device)
wandb.init(project='bias_free_modeling_english_discrete', name = 'golden')
train_model(model, optimizer, criterion, train_dataloader, val_dataloader, num_epochs, device, save_dir = SAVE_DIR,
            use_wandb = True)
