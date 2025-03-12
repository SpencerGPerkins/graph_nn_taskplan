import os
import json
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from graph_constructor_v3 import Graph
from MultiHead_GAT import MultiHeadGAT

# Load dataset and create graph instances
def load_dataset(vision_path, llm_path, label_path, num_data_samples):
    data_list = []
    for f in range(num_data_samples):
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
        encodings = graph.get_wire_encodings()
        print(type(encodings), encodings.layout)
        # Convert to PyG Data object
        wire_encodings = graph.get_wire_encodings()
        print(f"wire_encodings: {wire_encodings}")
        
        # padded_wire_encodings = [feat + torch.tensor([0]) * (13 - len(feat)) for feat in wire_encodings]
        terminal_encodings = graph.get_terminal_encodings()
        print(f"terminal encodings : {terminal_encodings}")
        # x = torch.tensor(list(wire_encodings) + list(terminal_encodings), dtype=torch.long)
        x = torch.cat([wire_encodings, terminal_encodings], dim=0)
        print(f"X : {x}")
        edge_index = graph.get_edge_index()
        print(f"Shape of edge index: {edge_index.shape}")
        y = graph.get_labels()
        print(f"Labels  : {y}")
        
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
        print(f"Processing data : {f}")
    
    return data_list

def train(model, loader, optimizer, wire_criterion, terminal_criterion, action_criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        wire_logits, terminal_logits, action_logits = model(
            data.x.float(), data.edge_index,
            num_wires=len(data.x) - 10,  # Assuming last 10 nodes are terminals
            num_terminals=10
        )

        # Unpack labels
        wire_label, terminal_label, action_label = data.y
        
        # For wire and terminal, apply BCEWithLogitsLoss (binary classification)
        wire_loss = wire_criterion(wire_logits, wire_label.float())  # wire_label should be 0 or 1
        terminal_loss = terminal_criterion(terminal_logits, terminal_label.float())  # terminal_label should be 0 or 1

        # For action, apply CrossEntropyLoss (multi-class classification)
        action_loss = action_criterion(action_logits, action_label)

        # Total loss
        loss = wire_loss + terminal_loss + action_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # For action classification, get predicted class
        all_preds.extend(action_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(action_label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), acc, f1

def validate(model, loader, wire_criterion, terminal_criterion, action_criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            wire_logits, terminal_logits, action_logits = model(
                data.x.float(), data.edge_index, 
                num_wires=len(data.x) - 10,
                num_terminals=10
            )

            # Unpack labels
            wire_label, terminal_label, action_label = data.y
            
            # For wire and terminal, apply BCEWithLogitsLoss (binary classification)
            wire_loss = wire_criterion(wire_logits, wire_label.float())  # wire_label should be 0 or 1
            terminal_loss = terminal_criterion(terminal_logits, terminal_label.float())  # terminal_label should be 0 or 1

            # For action, apply CrossEntropyLoss (multi-class classification)
            action_loss = action_criterion(action_logits, action_label)

            # Total loss
            total_loss += wire_loss.item() + terminal_loss.item() + action_loss.item()
            
            # For action classification, get predicted class
            all_preds.extend(action_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(action_label.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), acc, f1

def main():
    # Main script
    vision_data = f"../synthetic_data/vision/"
    llm_data = f"../synthetic_data/llm/"
    label_data = f"../synthetic_data/labels/"
    dataset = load_dataset(vision_data, llm_data, label_data, num_data_samples=500)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    train_data, val_data = dataset[:train_size], dataset[train_size:]

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiHeadGAT(in_dim=len(dataset[0].x[0]), hidden_dim=16, max_wires=10000, max_terminals=10, num_actions=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    binary_criterion = torch.nn.BCELoss()
    action_criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(20):
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, binary_criterion, binary_criterion, action_criterion, device=device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, binary_criterion, binary_criterion, action_criterion, device=device)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}, F1 {train_f1:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")
        
        
if __name__ == "__main__":
    main()


