import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class TaskGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, object_classes, goal_classes, action_classes):
        super(TaskGNN, self).__init__()
        
        # Graph Attention Layers
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        # Multi-head classification
        self.object_head = torch.nn.Linear(hidden_dim, object_classes)  # Wire selection
        self.goal_head = torch.nn.Linear(hidden_dim, goal_classes)      # Terminal selection
        self.action_head = torch.nn.Linear(hidden_dim, action_classes)  # Action selection
    
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Pooling to get graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Multi-head outputs
        object_pred = F.softmax(self.object_head(x), dim=1)
        goal_pred = F.softmax(self.goal_head(x), dim=1)
        action_pred = F.softmax(self.action_head(x), dim=1)
        
        return object_pred, goal_pred, action_pred  # Three classification heads

# Example usage
input_dim = 5   # Feature vector size per node
hidden_dim = 16 # Hidden layer size
object_classes = 3  # Example: 3 different wires
goal_classes = 5    # Example: 5 terminals
action_classes = 4  # Actions: Pick, Place, Insert, Lock Screw

gnn = TaskGNN(input_dim, hidden_dim, object_classes, goal_classes, action_classes)

# Example Graph Data (Random)
nodes = torch.rand((8, input_dim))  # 8 nodes with random features
edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],  # Edge connections
                       [1, 2, 3, 4, 5, 6, 7, 0]], dtype=torch.long)
batch = torch.zeros(8, dtype=torch.long)  # Single graph in batch

# Forward pass
object_probs, goal_probs, action_probs = gnn(nodes, edges, batch)
print("Object probabilities:", object_probs)
print("Goal probabilities:", goal_probs)
print("Action probabilities:", action_probs)

# Training Function
def train_gnn(model, dataloader, optimizer, loss_fn, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            obj_pred, goal_pred, act_pred = model(data.x, data.edge_index, data.batch)
            
            loss_obj = loss_fn(obj_pred, data.y_object)
            loss_goal = loss_fn(goal_pred, data.y_goal)
            loss_action = loss_fn(act_pred, data.y_action)
            
            loss = loss_obj + loss_goal + loss_action  # Multi-task loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Example Training Data (Simulated Expert Demonstrations)
data_list = [Data(x=torch.rand((8, input_dim)), edge_index=edges, batch=batch, 
                  y_object=torch.randint(0, object_classes, (1,)),
                  y_goal=torch.randint(0, goal_classes, (1,)),
                  y_action=torch.randint(0, action_classes, (1,))) for _ in range(50)]

dataloader = DataLoader(data_list, batch_size=8, shuffle=True)

# Training
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
train_gnn(gnn, dataloader, optimizer, loss_fn)
