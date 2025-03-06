import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, accuracy_score
from graph_constructorV2 import Graph

# Load dataset and create graph instances
def load_dataset(vision_path, llm_path, label_path):
    data_list = []
    for f in range(500):
        vision_data = f"{vision_path}sample_{f}.json"
        llm_data = f"{llm_path}sample_{f}.json"
        label_data = f"{label_path}sample_{f}.json"
        
        # Create Graph instance
        graph = Graph(
            vision_in=vision_data,
            llm_in=llm_data,
            label_in=label_data
        )
        graph.gen_encodings()
        
        # Convert to PyG Data object
        wire_encodings = graph.get_wire_encodings().values()
        
        padded_wire_encodings = [feat + [0] * (13 - len(feat)) for feat in wire_encodings]
        terminal_encodings = graph.get_terminal_encodings().values()
        x = torch.tensor(list(padded_wire_encodings) + list(terminal_encodings), dtype=torch.long)
        edge_index = graph.get_edge_index()
        edge_attr = graph.get_edge_features()
        y = graph.get_labels()
        
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        print(f"Processing data : {f}")
    
    return data_list

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_wires, max_terminals, num_actions, heads=4):
        super(MultiHeadGAT, self).__init__()
        
        # GAT layers with edge attention
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=2, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=2, concat=False)
        
        # Output heads with max sizes
        self.wire_head = nn.Linear(hidden_dim, max_wires)
        self.terminal_head = nn.Linear(hidden_dim, max_terminals)
        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, edge_index, edge_attr, num_wires, num_terminals):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        
        wire_logits = self.wire_head(x)[:, :num_wires]  # Only use relevant logits
        terminal_logits = self.terminal_head(x)[:, :num_terminals]
        action_logits = self.action_head(x)
        
        return wire_logits, terminal_logits, action_logits

# Train function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for data in loader:
        optimizer.zero_grad()
        out = model(data, )
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1

# Validation function
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1

# Main script
vision_data = f"../synthetic_data/vision/"
llm_data = f"../synthetic_data/llm/"
label_data = f"../synthetic_data/labels/"

data_list = load_dataset(vision_data, llm_data, label_data)
dataset_size = len(data_list)
train_size = int(0.8 * dataset_size)
train_data, val_data = data_list[:train_size], data_list[train_size:]

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

model = MultiHeadGAT(in_dim=len(data_list[0].x[0]), hidden_dim=16, max_wires=10000, max_terminals=10, num_actions=3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}, F1 {train_f1:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")
