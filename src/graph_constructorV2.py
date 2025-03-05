import torch
import networkx as nx 
import matplotlib.pyplot as plt
import json

class Graph:
    
    def __init__(self, vision_in, llm_in, label_in=None):
        
        """
        Args
        -------
        vision_in : json file, detected objects / states
        llm_in : json file, target wire and goal state 
        label_in : json file, conditional, only if training
        """
        
        # Predefined data
        terminals = [
            "terminal_0",
            "terminal_1", "terminal_2",
            "terminal_3", "terminal_4",
            "terminal_5", "terminal_6",
            "terminal_7", "terminal_8",
            "terminal_9"
        ]
        colors = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white"
        ]
        
        # Process input data
        with open(vision_in, 'r') as vision_file:
            vision_data = json.load(vision_file)
        with open(llm_in, 'r') as llm_file:
            llm_data = json.load(llm_file)
        if label_in != None:
            with open(label_in, 'r') as label_file:
                label_data = json.load(label_file)
                
        goal_states = []
        goal_positions = []
        
        # Process predefined data
        for terminal in terminals:

            print(type(vision_data["terminals"][terminal]["state"]))
            goal_states.append(vision_data["terminals"][terminal]["state"])
            goal_positions.append(vision_data["terminals"][terminal]["coordinates"])
            
        # Initialize data

        self.detected_wires = [wire["name"] for wire in vision_data["wires"]]
        self.wire_positions = [wire["coordinates"] for wire in vision_data["wires"]]
        self.wire_states = [wire["state"] for wire in vision_data["wires"]]
        self.terminals = terminals
        self.terminal_states = goal_states
        self.terminal_positions = goal_positions
        
        self.target_wire = llm_data["target_wire"]
        self.target_terminal = llm_data["target_terminal"]
        
        self.node_positions = self.wire_positions + self.terminal_positions

        # Init encodings and graph structure
        self.wire_encodings = {}
        self.terminal_encodings = {}
        self.edge_index = None
        self.adj_matrix = None
        self.edge_features = None
        
    def gen_encodings(self):
        # Generate encodings and graph
        self.node_feature_encoding()
        self.edge_index_adj_matrix()
        self.edge_features_encoding(self.target_wire, self.target_terminal)   
    
    def one_hot_encode(self, value, categories):
        encoding = [0] * len(categories)
        encoding[categories.index(value)] = 1
        
        return encoding
        
    def euclidean_distance(self, pos1, pos2):
        print()
        if type(pos1) == list and type(pos2) == list:
            pos1 = torch.tensor(pos1, dtype=torch.float32)  # Convert list to tensor
            pos2 = torch.tensor(pos2, dtype=torch.float32)  # Convert list to tensor
            return torch.norm(pos1 - pos2, p=2).item()
        else:
            return torch.norm(pos1 - pos2, p=2).item()

    
    def node_feature_encoding(self):
        # Encode wires
        for w, wire in enumerate(self.detected_wires):
            color, _ = wire.split("_")
            wire_features = self.one_hot_encode(wire, self.detected_wires)
            self.wire_encodings[f"wire_{w}"] = wire_features
            
        # Encode terminals
        for t, terminal in enumerate(self.terminals):
            terminal_features = self.one_hot_encode(terminal, self.terminals) + self.one_hot_encode(self.terminal_states[t], ["empty", "inserted", "locked"])    
            self.terminal_encodings[terminal] = terminal_features
            
    def edge_index_adj_matrix(self):
        # Combine objects and terminals into one list of nodes
        nodes = self.detected_wires + self.terminals
        num_nodes = len(nodes)        
        
         # Create edge index (fully connected Graph)
        edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        
        # Convert edge_index to tensor
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create Adjacency matrix
        self.adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)

    def edge_features_encoding(self, target_object, target_goal):
        edge_features = []
        
        for i in range(self.edge_index.shape[1]):
            src, tgt = self.edge_index[:, i] # Get source and target nodes
            wire_term_dict = {
                "wires": [],
                "terminals": []
            }
            
            edge_index_list = self.edge_index.tolist()
            for item in edge_index_list[0]:
                if item <= len(self.detected_wires)-1:
                    wire_term_dict["wires"].append(item)
                elif item > len(self.detected_wires)-1:
                    wire_term_dict["terminals"].append(item)
            if src in wire_term_dict["wires"]:
                importance = 5.0 if (self.detected_wires[src] in target_object and self.terminals[tgt-len(self.detected_wires)] in target_goal) else 1.0
            elif src in wire_term_dict["terminals"] and tgt in wire_term_dict["wires"]:
                importance = 5.0 if (self.detected_wires[tgt] in target_object and self.terminals[src-len(self.detected_wires)] in target_goal) else 1.0
            else:
                importance = 1.0
                 
            print(f"\nProcessing edge {i} - source: {src}, target: {tgt}\n")

            # print(self.node_positions[src], self.node_positions[tgt])
            distance = self.euclidean_distance(self.node_positions[src], self.node_positions[tgt])
            if importance == 5.0:
                print(f"Target wire: {target_object}, Target Terminal: {target_goal}\n")
                print(f"Source Node: {src}, Target Node: {tgt}\n")
            edge_features.append([distance, importance])
               
            print(f"\nEdge {i} features appended: Distance={distance}, Importance={importance}\n")

        # Convert edge features to tensor
        self.edge_features = torch.tensor(edge_features, dtype=torch.float)        
    
    def get_wire_encodings(self):
        return self.wire_encodings
    
    def get_terminal_encodings(self):
        return self.terminal_encodings
    
    def get_edge_index(self):
        return self.edge_index
    
    def get_adj_matrix(self):
        return self.adj_matrix
    
    def get_edge_features(self):
        return self.edge_features       
                
                
                           
def visualize_graph_with_features(graph, num):

    G = nx.DiGraph() # Undirected Graph
    
    # Add nodes (Objects +Goals)
    node_labels = {} # Labels for annotation
    colors = [] # Node Colors
    
    num_objects = len(graph.detected_wires)
    num_goals = len(graph.terminals)
    
    for o, obj in enumerate(graph.detected_wires):
        G.add_node(o, pos=graph.node_positions[o])
        node_labels[o] = f"{obj}\n{graph.wire_encodings['wire_' +str(o)]}" # Show node features
        colors.append("red") # Wire node color
        
    for g, goal in enumerate(graph.terminals):
        goal_idx = num_objects + g 
        G.add_node(goal_idx, pos=graph.node_positions[goal_idx])
        node_labels[goal_idx] = f"{goal}\n{graph.terminal_encodings[goal]}" # Show node features
        colors.append("blue") # Terminal node color
    
    # Add edges with feature-based weights
    edges = graph.get_edge_index().t().tolist()
    edge_labels = {} # Store edge labels
    edge_weights = [] # Store edge weights for visualization
    edge_color = []
    edge_features = graph.get_edge_features()
    
    for i, (src, tgt) in enumerate(edges):
        distance, importance = edge_features[i]

        G.add_edge(src, tgt)
          
        edge_labels[(src, tgt)] = f"D:{distance:.2f}, W:{importance}" # Distance and Weight
        edge_weights.append(importance) # Importance as thickness
        if importance == 5.0:
            edge_color.append("red")
            print(f"Visualize Source Node: {src}, Target Node: {tgt}\n")

        else:
            edge_color.append("gray")

    # Node postitions
    pos = nx.kamada_kawai_layout(G) # Graph layout
    # pos = nx.spring_layout(G)

    # Draw graph
    plt.figure(figsize=(20, 17))
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=node_labels,
        edge_color=edge_color, node_size=4500, font_size=8, width=3.0)
    
    plt.title("Graph with Node & Edge Features")
    plt.savefig(f"../docs/graph_gen_example_sample{num}.pdf")
    plt.show()    
    

for i in range(50):    
    # Input path
    vision_data_path = f'../synthetic_data/vision/sample_{i}.json'
    llm_data_path = f'../synthetic_data/llm/sample_{i}.json'
    y_data_path = f'../synthetic_data/labels/sample_{i}.json'

    graph = Graph(vision_data_path, llm_data_path)
    graph.gen_encodings()

    # Call visualization function
    visualize_graph_with_features(graph, num=i)
