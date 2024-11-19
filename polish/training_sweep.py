import sys
sys.path.append(r'D:\GitHub\bias_free_modeling')
from polish.model import SPropGNN
from polish.dataset import SPropDataset
from polish.utils import load_data
from torch_geometric.loader import DataLoader
from general_utils import set_seed
import torch
import torch.nn as nn
import torch.optim as optim
from polish.training_loop import train_model
import os
import pickle
import wandb


def main():


    sweep_configuration = {
        "name": "gnn_sweep_final",
        "entity" : 'hubertp',
        "metric": {"name": "best_corr", "goal": "maximize"},
        "method": "bayes",
        "parameters": {"dropout": {"values": [0, 0.2, 0.4, 0.6]},
                       "learning_rate": {"values": [5e-3, 5e-4, 5e-5]},
                       "weight_decay": {"values": [0.0, 0.2, 0.4, 0.6]}}
    }


    sweep_id = wandb.sweep(sweep_configuration, project="bias_free_modeling_polish_dimensions")

    train, test, val = load_data()
    redo = False
    if os.path.exists(r'D:\GitHub\bias_free_modeling\data/polish/train_hierarchical.pt') and redo == False:
        train_dataset = torch.load(r'D:\GitHub\bias_free_modeling\data/polish/train_hierarchical.pt')
        val_dataset = torch.load(r'D:\GitHub\bias_free_modeling\data/polish/val_hierarchical.pt')
    else:
        train_dataset, val_dataset = SPropDataset(train), SPropDataset(val)
        torch.save(train_dataset, r'D:\GitHub\bias_free_modeling\data/polish/train_hierarchical.pt')
        torch.save(val_dataset, r'D:\GitHub\bias_free_modeling\data/polish/val_hierarchical.pt')

    batch = 400

    epochs = 200

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

    with open(r'D:\GitHub\bias_free_modeling\data/polish/pos_tags.pkl', 'rb') as f:
        pos_tags = pickle.load(f)

    with open(r'D:\GitHub\bias_free_modeling\data/polish/parser_categories.pkl', 'rb') as f:
        parser_tag_categories = pickle.load(f)



    def train(config=None):

      with wandb.init(config = config):
        config = wandb.config
        torch.cuda.empty_cache()

        metric_names = ['norm_Valence_M', 'norm_Arousal_M']
        num_node_features = 3  # Valence Arousal and sentence location

        model = SPropGNN(num_node_features, len(pos_tags) + 1, len(parser_tag_categories) + 2, metric_names,
                             dropout_prob=config.dropout)

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.MSELoss()

        model = model.to(device)
        criterion = criterion.to(device)


        train_model(model, optimizer, criterion, train_dataloader, val_dataloader, epochs, metric_names, device,
                    save_dir = None, use_wandb = True)


        del model, criterion


    wandb.agent(sweep_id, train, count=20)

if __name__ == '__main__':
    main()