import torch
import networkx as nx 
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, object_types, object_states, goal_states, node_positions, target_wire, target_terminal):
        # Init data
        self.object_types = object_types
        self.object_states = object_states
        self.goal_states = goal_states
        self.node_positions = node_positions
        self.target_wire = []
        self.target_terminal = []
        
        # Find target object detection index
        for o, object in enumerate(self.object_types):
            if object == target_wire:
                self.target_wire.append(o)
        
        # Pre-defined categories
        self.color_types =[
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white"
        ]
        self.goal_types = [
            "terminal_0",
            "terminal_1", "terminal_2",
            "terminal_3", "terminal_4",
            "terminal_5", "terminal_6",
            "terminal_7", "terminal_8",
            "terminal_9"
        ]

        # Find target terminal index
        for t, terminal in enumerate(self.goal_types):
            if terminal == target_terminal:
                self.target_terminal.append(t+(len(self.object_types)))
        # print(f"look up target terminals: {self.target_terminal}")
                
        # Init encodings and graph structure
        self.wire_encodings = {}
        self.terminal_encodings = {}
        self.edge_index = None
        self.adj_matrix = None
        self.edge_features = None
    #----------------------------------EDITING-----------------------#
    
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
        return torch.norm(pos1 - pos2, p=2).item()
    
    def node_feature_encoding(self):
        # Encode wires
        for w, wire in enumerate(self.object_types):
            color, _ = wire.split("_")
            wire_features = self.one_hot_encode(wire, self.object_types)
            self.wire_encodings[f"wire_{w}"] = wire_features
            
        # Encode terminals
        for t, terminal in enumerate(self.goal_types):
            terminal_features = self.one_hot_encode(terminal, self.goal_types) + self.goal_states[t]
            self.terminal_encodings[terminal] = terminal_features
            
    def edge_index_adj_matrix(self):
        # Combine objects and terminals into one list of nodes
        nodes = self.object_types + self.goal_types
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
            # print(self.edge_index[:,i])
            src, tgt = self.edge_index[:, i] # Get source and target nodes
            distance = self.euclidean_distance(self.node_positions[src], self.node_positions[tgt])
            # print(f"CHECK TARGETS: {target_object, target_goal}")
            # Task importance: Assign higher weight if edge connects LLM target wire and target terminal
            importance = 5.0 if (src in target_object and tgt in target_goal) or (tgt in target_object and src in target_goal) else 1.0
            if importance == 5.0:
                print(f"Target wire: {target_object}, Target Terminal: {target_goal}\n")
                print(f"Source Node: {src}, Target Node: {tgt}\n")
            edge_features.append([distance, importance])
            
        
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
    

def visualize_graph_with_features(graph):

    G = nx.DiGraph() # Undirected Graph
    
    # Add nodes (Objects +Goals)
    node_labels = {} # Labels for annotation
    colors = [] # Node Colors
    
    num_objects = len(graph.object_types)
    num_goals = len(graph.goal_types)
    
    for o, obj in enumerate(graph.object_types):
        G.add_node(o, pos=graph.node_positions[o].tolist())
        node_labels[o] = f"{obj}\n{graph.wire_encodings['wire_' +str(o)]}" # Show node features
        colors.append("red") # Wire node color
        
    for g, goal in enumerate(graph.goal_types):
        goal_idx = num_objects + g 
        G.add_node(goal_idx, pos=graph.node_positions[goal_idx].tolist())
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
        # print(f"source and target: {(src, tgt)}")

        G.add_edge(src, tgt)
        # print(f"labels: {(distance, importance)}")    
        edge_labels[(src, tgt)] = f"D:{distance:.2f}, W:{importance}" # Distance and Weight
        edge_weights.append(importance) # Importance as thickness
        if importance == 5.0:
            # edge_weights.append(10.0)
            edge_color.append("red")
            print(f"Visualize Source Node: {src}, Target Node: {tgt}\n")
            # print(edge_weights[i])
        else:
            # edge_weights.append(0.25)
            edge_color.append("gray")

    # Node postitions
    pos = nx.kamada_kawai_layout(G) # Graph layout
    # pos = nx.spring_layout(G)

    # Draw graph
    plt.figure(figsize=(20, 17))
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=node_labels,
        edge_color=edge_color, node_size=4500, font_size=8, width=3.0)
    
    # # Draw edge labels
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="green")
    
    plt.title("Graph with Node & Edge Features")
    plt.savefig("./docs/graph_gen_example.pdf")
    plt.show()
    
    
    
#------------------------------------CALL Functions, Synthetic DATA-----------------------------------#


# Inputs
# objects = ["red_wire", "blue_wire", "yellow_wire", "red_wire", "yellow_wire", "yellow_wire"]
# object_states = [[1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0]]
objects = ["red_wire", "blue_wire"]
object_states = [[1,0,0], [1,0,0]]
# objects = ["red_wire", "blue_wire", "red_wire"]
# object_states = [[1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0], [1,0,0]]
goal_states = [[1,0]] * 9 
node_positions = torch.randn(len(objects) + len(goal_states), 2)

# Create graph
graph = Graph(objects, object_states, goal_states, node_positions, target_wire="blue_wire", target_terminal="terminal_1")
graph.gen_encodings() # NOT SURE IF I WANT THIS AS A SEPARATE FUNCTION

# Accessing encoded features
# print(f"Wire encodings:\n{graph.get_wire_encodings()}")
# print(f"Terminal encodings:\n{graph.get_terminal_encodings()}")

# # Accessing graph structure
# print(f"Edge index:\n{graph.get_edge_index()}")
# print(f"Adjacency Matrix:\n{graph.get_adj_matrix()}")

# # Accessing edge features
# print(f"Edge Features:\n{graph.get_edge_features()}")

# Call visualization function
visualize_graph_with_features(graph)
    
        