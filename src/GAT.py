import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_wires, num_terminals, num_actions, heads=4):
        super(MultiHeadGAT, self).__init__()
        
        # GAT layers with edge attention
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=2, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=2, concat=False)
        
        # Output heads for classification
        self.wire_head = nn.Linear(hidden_dim, num_wires)  # Predicts target wire
        self.terminal_head = nn.Linear(hidden_dim, num_terminals)  # Predicts target terminal
        self.action_head = nn.Linear(hidden_dim, num_actions)  # Predicts correct action
        
    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        
        wire_logits = self.wire_head(x)
        terminal_logits = self.terminal_head(x)
        action_logits = self.action_head(x)
        
        return wire_logits, terminal_logits, action_logits


def train_gat(model, data, epochs=100, lr=0.005):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        wire_logits, terminal_logits, action_logits = model(
            data.x, data.edge_index, data.edge_attr
        )
        
        loss_wire = criterion(wire_logits, data.y_wire)
        loss_terminal = criterion(terminal_logits, data.y_terminal)
        loss_action = criterion(action_logits, data.y_action)
        
        loss = loss_wire + loss_terminal + loss_action  # Multi-task loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model
