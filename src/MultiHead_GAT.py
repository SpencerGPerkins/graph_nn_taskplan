import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_wires, max_terminals, num_actions, heads=4):
        super(MultiHeadGAT, self).__init__()
        
        # GAT layers with edge attention
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=2, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=2, concat=False)
        
        # Output heads for binary classification (sigmoid output for binary classification)
        self.wire_head = nn.Linear(hidden_dim, 1)  # Binary classification, single output per node
        self.terminal_head = nn.Linear(hidden_dim, 1)  # Binary classification, single output per node
        self.action_head = nn.Linear(hidden_dim, num_actions)  # For actions, keep it as is (multi-class classification)

    def forward(self, x, edge_index, num_wires, num_terminals):
        # Apply GAT layers with edge attention
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        # Wire head: Binary classification (sigmoid activation)
        wire_logits = self.wire_head(x)[:, :num_wires]  # Only use relevant logits
        wire_probs = torch.sigmoid(wire_logits)  # Apply sigmoid for binary classification
        
        # Terminal head: Binary classification (sigmoid activation)
        terminal_logits = self.terminal_head(x)[:, :num_terminals]
        terminal_probs = torch.sigmoid(terminal_logits)  # Apply sigmoid for binary classification
        
        # Action head: Multi-class classification (softmax or other appropriate activation)
        action_logits = self.action_head(x)
        
        return wire_probs, terminal_probs, action_logits

# class MultiHeadGAT(nn.Module):
#     def __init__(self, in_dim, hidden_dim, max_wires, max_terminals, num_actions, heads=4):
#         super(MultiHeadGAT, self).__init__()
        
#         # GAT layers with edge attention
#         self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=2, concat=True)
#         self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=2, concat=False)
        
#         # Output heads with max sizes
#         self.wire_head = nn.Linear(hidden_dim, max_wires)
#         self.terminal_head = nn.Linear(hidden_dim, max_terminals)
#         self.action_head = nn.Linear(hidden_dim, num_actions)

#     def forward(self, x, edge_index, num_wires, num_terminals):
#         x = F.elu(self.gat1(x, edge_index))
#         x = F.elu(self.gat2(x, edge_index))
        
#         wire_logits = self.wire_head(x)[:, :num_wires]  # Only use relevant logits
#         terminal_logits = self.terminal_head(x)[:, :num_terminals]
#         action_logits = self.action_head(x)
        
#         return wire_logits, terminal_logits, action_logits