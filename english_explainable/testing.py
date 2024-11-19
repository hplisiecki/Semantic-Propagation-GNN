from english_explainable.model import SPropGNN
from english.dataset import SPropDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import pickle

# import pearson
from scipy.stats import pearsonr

# train, test, val = load_data()

# test_dataset = SPropDataset(test)

# save
# torch.save(test_dataset, r'data/english/test_hierarchical.pt')

# # load
test_dataset = torch.load(r'data/english/test_hierarchical.pt')

test_dataloader = DataLoader(test_dataset, batch_size=400, shuffle=False, drop_last=True)
predicted_metrics = ['norm_Valence_M', 'norm_Arousal_M']
metric_names = ['norm_Valence_M', 'norm_Arousal_M']

batch = 400
dropout = 0.6
learning_rate = 0.00005
weight_decay = 0.4



with open(r'data/english/pos_tags.pkl', 'rb') as f:
    pos_tags = pickle.load(f)

with open(r'data/english/parser_tag_categories.pkl', 'rb') as f:
    parser_tag_categories = pickle.load(f)

num_node_features = 3 # POS tag, Negation, Valence, Arousal, Dominance
model = SPropGNN(num_node_features, len(pos_tags) + 1, len(parser_tag_categories) + 2, metric_names,
                     dropout_prob=dropout)
# load
SAVE_DIR = r'D:\GitHub\bias_free_modeling\models\scalar_scaling_english_dimensions'


model.load_state_dict(torch.load(SAVE_DIR))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

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


from general_utils import visualize_scaling_factors, visualize_scaling_factors_simple


text = 'I am not happy.'
import pandas as pd

df_text = pd.DataFrame({'text': [text], 'norm_Valence_M': [0.5], 'norm_Arousal_M': [0.5], 'norm_Dominance_M': [0.5]})
text_dataset = SPropDataset(df_text)
text_dataloader = DataLoader(text_dataset, batch_size=1, shuffle=False, drop_last=True)
model.eval()
model = model.cuda()

total_val_loss = 0
all_preds = {metric: [] for metric in predicted_metrics}
all_labels = {metric: [] for metric in predicted_metrics}
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


# Access activations
activations = model.activations

# Visualize input features
# features = visualize_input_features(data)

# Visualize scaling factors for the first GNN layer
visualize_scaling_factors(data, activations, layer_index=0, save = 'plots/happy2.svg')

visualize_scaling_factors_simple(data, activations, layer_index=0, save = 'plots/nothappy.svg', seed = 32,
                                 title = None, x_off = 0, y_off = 0)

# Valence correlation: 0.6235581067345796
# Arousal correlation: 0.4523542886098063
# Validation loss: 0.0030310360714793207
