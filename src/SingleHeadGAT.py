import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class SingleHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_wires, max_terminals, num_actions, heads=4):
        super(SingleHeadGAT, self).__init__()
        
        # GAT layers with edge attention
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=2, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=2, concat=False)
        # Global pooling layer (e.g., mean pooling)
        self.global_pool = global_mean_pool  # You can also try global_max_pool

        self.action_head = nn.Linear(hidden_dim, num_actions)  # For actions, keep it as is (multi-class classification)

    def forward(self, x, edge_index, batch, num_wires, num_terminals):
        # Apply GAT layers with edge attention
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = self.global_pool(x, batch)  # Shape: [num_graphs, hidden_dim]

        # Action head: Multi-class classification (softmax or other appropriate activation)
        action_logits = self.action_head(x)
        print(f"Innetwork action logits: {action_logits}")
        
        return action_logits
