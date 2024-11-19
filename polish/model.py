import torch.nn as nn
import torch.nn.functional as F
from custom_gnn.custom_layers import SPropConv, CustomGlobalAttention
import torch


class GNNLayerSequence(nn.Module):
    """
    A sequential container for GNN layers, which applies each layer in sequence and applies a ReLU activation after each layer's output
    """
    def __init__(self, layers):
        super(GNNLayerSequence, self).__init__()
        # Initialize with a list of layers to be applied sequentially
        self.layers = nn.ModuleList(layers)

    def forward(self, x, node_types, edge_index, edge_type):
        # Apply each layer in sequence
        for layer in self.layers:
            # If layer is an SPropConv, pass node and edge types along with features
            if isinstance(layer, SPropConv):
                x = layer(x, node_types, edge_index, edge_type)
            else:
                # For other layers, only pass node features
                x = layer(x)
            # Apply ReLU activation after each layer's output
            x = F.relu(x)
        return x


class SPropGNN(nn.Module):
    """
    The main model class for the SPropGNN model for continuous metric prediction, which consists of a sequence of GNN layers followed by
    a custom attention pooling layer and separate output layers for each metric.

    Parameters:
    - num_node_features (int): The number of input features for each node
    - num_node_types (int): The number of unique node types (parts of speech types + the sentence type) in the graph
    - num_relations (int): The number of unique edge types (dependency types + sentence relation type) in the graph
    - metric_names (list of str): The names of the metrics to predict
    - dropout_prob (float): The probability of dropout to apply to the output of the model
    """
    def __init__(self, num_node_features, num_node_types, num_relations, metric_names, dropout_prob=0.3):
        super(SPropGNN, self).__init__()

        # Initialize number of relations and metrics
        self.num_relations = num_relations
        self.metric_names = metric_names

        # Define a sequence of GNN layers with a single SPropConv layer
        self.relational_graph = GNNLayerSequence([
            SPropConv(num_node_features, 512),
        ])

        # Initialize node type and edge type embeddings
        self.node_type_embedding = nn.Embedding(num_node_types, 512)
        self.edge_type_embedding = nn.Embedding(num_relations, 512)

        # Define a custom attention pooling layer with a gate neural network
        self.attention_pool = CustomGlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ))

        # Define separate output layers for each metric, with dropout for regularization
        self.metric_outputs = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1024, 100),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(100, 1),
                nn.Sigmoid() # Apply sigmoid activation to ensure output is in [0, 1] range
            ) for name in metric_names
        })

    def forward(self, data):
        # Unpack input data
        x, node_types, edge_index, batch, edge_type = (
            data.x,
            data.node_type,
            data.edge_index,
            data.batch,
            data.edge_type,
        )

        # Store initial activations and input features for visualization or analysis
        self.activations = {
            'input_features': x.detach().cpu(),
            'node_types': node_types.detach().cpu(),
            'edge_index': edge_index.detach().cpu(),
            'edge_type': edge_type.detach().cpu(),
            'batch': batch.detach().cpu(),
        }

        # Generate node and edge embeddings
        node_type_embedding = self.node_type_embedding(node_types)
        edge_type_embedding = self.edge_type_embedding(edge_type)

        # Pass data through the relational graph layers
        x = self.relational_graph(x, node_type_embedding, edge_index, edge_type_embedding)

        # Store node embeddings for explainability
        self.activations['node_embeddings'] = x.detach().cpu()

        # Store scaling factors `s_ij` from each SPropConv layer for interpretability
        self.activations['s_ij_list'] = [layer.last_s_ij.detach().cpu() for layer in self.relational_graph.layers if
                                         isinstance(layer, SPropConv)]

        # Concatenate node features with their embeddings
        x_with_embeddings = torch.cat([x, node_type_embedding], dim=-1)

        # Apply attention pooling to obtain a graph-level embedding
        text_embedding = self.attention_pool(x_with_embeddings, data.batch)

        # Store attention weights for analysis
        self.activations['attention_weights'] = self.attention_pool.attention_weights

        # Apply output layers for each metric, generating predictions for each
        outputs = {name: self.metric_outputs[name](text_embedding) for name in self.metric_names}

        return outputs