import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import softmax


# Define the custom SPropConv layer, inheriting from PyTorch Geometric's MessagePassing class
class SPropConv(MessagePassing):
    def __init__(self, node_feature_dim, hidden_dim):
        # Initialize MessagePassing with 'add' aggregation method
        super(SPropConv, self).__init__(aggr='add')

        # Linear layer to transform input node features into the hidden dimension
        self.node_transform = nn.Linear(node_feature_dim, hidden_dim)

        # Linear layer to compute the transformation on concatenated features for messages
        self.linear_s = nn.Linear(hidden_dim * 4, hidden_dim)

        # ReLU activation for introducing non-linearity
        self.activation = nn.ReLU()

        # Initialize storage for scaling factors, used for explainability
        self.last_s_ij = None

    def forward(self, x, node_type_embedding, edge_index, edge_type_embedding):
        # Transform input node features into hidden dimension space
        h = self.node_transform(x)  # Shape: [num_nodes, hidden_dim]

        # Reset scaling factor storage at the start of each forward pass
        self.last_s_ij = None

        # Call propagate to send messages along edges
        # Pass transformed node features `h`, node type embeddings, and edge type embeddings
        out = self.propagate(edge_index, h=h, node_type_embedding=node_type_embedding,
                             edge_type_embedding=edge_type_embedding)
        return out

    def message(self, h_j, node_type_embedding_i, node_type_embedding_j, edge_type_embedding):
        # Concatenate `h_j` (neighbor features) with node and edge type embeddings for context
        combined = torch.cat([h_j, node_type_embedding_i, node_type_embedding_j, edge_type_embedding], dim=-1)

        # Transform the concatenated features with a tanh activation to obtain scaling factors `s_ij`
        s_ij = torch.tanh(self.linear_s(combined))

        # Compute message `m_ij` by scaling neighbor feature `h_j` by factor `s_ij`
        m_ij = s_ij * h_j

        # Store scaling factors for each edge (useful for debugging or model interpretability)
        self.last_s_ij = s_ij  # Shape: [num_edges, hidden_dim]

        return m_ij

    def update(self, aggr_out, h):
        # Apply activation on the combined aggregated message and initial node features
        h_new = self.activation(h + aggr_out)
        return h_new


# Define the custom SPropConv layer, inheriting from PyTorch Geometric's MessagePassing class
class SPropConvScalarScaling(MessagePassing):
    def __init__(self, node_feature_dim, hidden_dim):
        # Initialize MessagePassing with 'add' aggregation method
        super(SPropConvScalarScaling, self).__init__(aggr='add')

        # Linear layer to transform input node features into the hidden dimension
        self.node_transform = nn.Linear(node_feature_dim, hidden_dim)

        # Linear layer to compute the transformation on concatenated features for messages
        self.linear_s = nn.Linear(hidden_dim * 4, 1)

        # ReLU activation for introducing non-linearity
        self.activation = nn.ReLU()

        # Initialize storage for scaling factors, used for explainability
        self.last_s_ij = None

    def forward(self, x, node_type_embedding, edge_index, edge_type_embedding):
        # Transform input node features into hidden dimension space
        h = self.node_transform(x)  # Shape: [num_nodes, hidden_dim]

        # Reset scaling factor storage at the start of each forward pass
        self.last_s_ij = None

        # Call propagate to send messages along edges
        # Pass transformed node features `h`, node type embeddings, and edge type embeddings
        out = self.propagate(edge_index, h=h, node_type_embedding=node_type_embedding,
                             edge_type_embedding=edge_type_embedding)
        return out

    def message(self, h_j, node_type_embedding_i, node_type_embedding_j, edge_type_embedding):
        # Concatenate `h_j` (neighbor features) with node and edge type embeddings for context
        combined = torch.cat([h_j, node_type_embedding_i, node_type_embedding_j, edge_type_embedding], dim=-1)

        # Transform the concatenated features with a tanh activation to obtain scaling factors `s_ij`
        s_ij = torch.tanh(self.linear_s(combined))

        # Compute message `m_ij` by scaling neighbor feature `h_j` by factor `s_ij`
        m_ij = s_ij * h_j

        # Store scaling factors for each edge (useful for debugging or model interpretability)
        self.last_s_ij = s_ij  # Shape: [num_edges, hidden_dim]

        return m_ij

    def update(self, aggr_out, h):
        # Apply activation on the combined aggregated message and initial node features
        h_new = self.activation(h + aggr_out)
        return h_new

from torch_geometric.nn import global_add_pool


# Define a custom global attention pooling layer
class CustomGlobalAttention(nn.Module):
    def __init__(self, gate_nn, nn_module=None):
        super(CustomGlobalAttention, self).__init__()

        # Neural network to compute attention scores (gating function)
        self.gate_nn = gate_nn

        # Optional transformation applied to features after attention weighting
        self.nn_module = nn_module

        # Variable to store attention weights for each node
        self.attention_weights = None

    def forward(self, x, batch):
        # Compute attention scores for each node
        gate = self.gate_nn(x)  # Shape: [num_nodes, 1]

        # Normalize attention scores using softmax, within each graph in the batch
        gate = softmax(gate, index=batch)  # Shape: [num_nodes, 1]

        # Store attention weights for interpretability or visualization
        self.attention_weights = gate.detach().cpu()

        # Apply an optional transformation to node features
        if self.nn_module is not None:
            x = self.nn_module(x)

        # Scale features by their attention scores (element-wise multiplication)
        x = x * gate

        # Aggregate weighted node features to obtain graph-level features
        out = global_add_pool(x, batch)  # Shape: [batch_size, out_channels]

        return out
