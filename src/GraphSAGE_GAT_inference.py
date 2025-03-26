import torch
from GraphSAGE_GAT import GraphSAGE_GAT
from torch_geometric.data import Data
from graph_constructor_partial import GraphCategorical  
import json

# Load the trained model
device = torch.device("cpu")

possible_actions = ["pick", "insert", "lock", "putdown"]

data_saver = {
    "predicted_action":[],
    "previous_actions":[]
}
with open("../run_data/action_executions.json", "r") as in_file:
    history = json.load(in_file)

data_saver["previous_actions"] = history["previous_actions"]

# Load a new sample
vision_data = f"../run_data/vision/vision_to_gnn.json"
llm_data = f"../run_data/llm/llm_to_gnn.json"


# Create a graph instance
graph = GraphCategorical(vision_in=vision_data, llm_in=llm_data)
graph.gen_encodings()

# Convert to PyG Data object
x = torch.cat([graph.get_wire_encodings(), graph.get_terminal_encodings()], dim=0)
edge_index = graph.get_edge_index()

data = Data(x=x, edge_index=edge_index)  # Ensure format matches training
data = data.to(device)

checkpoint = torch.load("GraphSAGE_model_weights_4_class.pth", map_location="cpu")
model = GraphSAGE_GAT(
    in_dim=data.x.shape[1],  # Use the actual feature size
    hidden_dim=16, 
    max_wires=10000, 
    max_terminals=10, 
    num_actions=4
)

with torch.no_grad():
    model.load_state_dict(torch.load("GraphSAGE_model_weights_4_class.pth", map_location=device))
    model.to(device)
    model.eval()  

    action_logits = model(
        data.x.float(), data.edge_index, None  
        )
    
    print(action_logits)
    
    # Convert logits to predicted class
    predicted_action = action_logits.argmax(dim=1).cpu().numpy().tolist()

    data_saver["predicted_action"] = [possible_actions[predicted_action[-1]]]
    data_saver["previous_actions"].append(possible_actions[predicted_action[-1]])

    print(f"Predicted Action: {predicted_action}")

with open("../run_data/action_executions.json", "w") as file:
    json.dump(data_saver, file)