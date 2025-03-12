import torch
from SingleHeadGAT import SingleHeadGAT
from torch_geometric.data import Data
from graph_constructor_v3 import Graph  # As

import json

# Load the trained model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


data_saver = {
    "predicted_action":[],
    "ground_truth":[]
}

for d in range(50):
    # Load a new sample
    vision_data = f"../synthetic_testing_data/old_data/vision_test/test_sample_{d}.json"
    llm_data = f"../synthetic_testing_data/old_data/llm_test/test_sample_{d}.json"
    label_data = f"../synthetic_testing_data/old_data/labels_test/test_sample_{d}.json"

    # Create a graph instance
    graph = Graph(vision_in=vision_data, llm_in=llm_data, label_in=label_data)
    graph.gen_encodings()

    # Convert to PyG Data object
    x = torch.cat([graph.get_wire_encodings(), graph.get_terminal_encodings()], dim=0)
    edge_index = graph.get_edge_index()
    y = graph.get_labels()

    data = Data(x=x, edge_index=edge_index, y=y.unsqueeze(0))  # Ensure format matches training
    data = data.to(device)
    # model = SingleHeadGAT(
    #     in_dim=data.x.shape[1],  # Use the actual feature size
    #     hidden_dim=16, 
    #     max_wires=10000, 
    #     max_terminals=10, 
    #     num_actions=4
    # )
    checkpoint = torch.load("model_weights.pth", map_location="cpu")
    model = SingleHeadGAT(
        in_dim=data.x.shape[1],  # Use the actual feature size
        hidden_dim=16, 
        max_wires=10000, 
        max_terminals=10, 
        num_actions=3
    )

    # # Load only matching parameters
    # model_dict = model.state_dict()
    # # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    # # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)

    # Initialize the new class weights randomly
    with torch.no_grad():
        # model.action_head.weight[-1] = torch.randn_like(model.action_head.weight[0])
        # model.action_head.bias[-1] = torch.tensor(0.0)  # Or another initialization strategy
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
        model.to(device)
        model.eval()  # Set model to evaluation mode
        # Run the model on the new sample
        # with torch.no_grad():
        action_logits = model(
            data.x.float(), data.edge_index, None,  # Assuming batch=None for single sample
            num_wires=len(data.x) - 10,  # Assuming last 10 nodes are terminals
            num_terminals=10
            )
        print(action_logits)
        # Convert logits to predicted class
        predicted_action = action_logits.argmax(dim=1).cpu().numpy().tolist()
        data_saver["predicted_action"].append(predicted_action)
        true_action = data.y.argmax(dim=1).cpu().numpy().tolist()
        data_saver["ground_truth"].append(true_action)
        print(f"True Action: {true_action}")

        print(f"Predicted Action: {predicted_action}")

with open("../docs/testing_results.json", "w") as file:
    json.dump(data_saver, file)
    

    