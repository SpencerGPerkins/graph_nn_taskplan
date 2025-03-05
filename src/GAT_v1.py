import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import f1_score, accuracy_score
from graph_constructor import Graph

# Load dataset and create graph instances
def load_dataset(data_dir):
    data_list = []
    for filename in os.listdir(os.path.join(data_dir, "vision")):
        with open(os.path.join(data_dir, "vision", filename), "r") as f:
            vision_data = json.load(f)
        with open(os.path.join(data_dir, "llm", filename), "r") as f:
            llm_data = json.load(f)
        with open(os.path.join(data_dir, "labels", filename), "r") as f:
            labels_data = json.load(f)

        # Create Graph instance
        graph = Graph(
            vision_data["object_types"], vision_data["object_states"],
            vision_data["goal_states"], vision_data["node_positions"],
            llm_data["target_wire"], llm_data["target_terminal"]
        )
        graph.gen_encodings()
        
        # Convert to PyG Data object
        x = torch.tensor(list(graph.get_wire_encodings().values()) + list(graph.get_terminal_encodings().values()), dtype=torch.float)
        edge_index = graph.get_edge_index()
        edge_attr = graph.get_edge_features()
        y = torch.tensor(labels_data["action"], dtype=torch.long)
        
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    
    return data_list

# Define GAT model
class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=2):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, edge_dim=2)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, edge_dim=2)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x

# Train function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
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
data_list = load_dataset("../synthetic_data")
dataset_size = len(data_list)
train_size = int(0.8 * dataset_size)
train_data, val_data = data_list[:train_size], data_list[train_size:]

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

model = GATModel(in_dim=len(data_list[0].x[0]), hidden_dim=16, out_dim=3).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}, F1 {train_f1:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")
