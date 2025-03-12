import networkx as nx 
import matplotlib.pyplot as plt
import datetime
import os


month = datetime.datetime.now().month
day = datetime.datetime.now().day
year = datetime.datetime.now().year
hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute

import networkx as nx
import matplotlib.pyplot as plt
import os

def visualize_graph(graph):
    global month, day, year, hour, minute

    G = nx.Graph()
    
    # Initialize node labels and colors
    node_labels = {}
    colors = []

    # Add wire nodes
    for w, wire in enumerate(graph.detected_wires):
        G.add_node(w, pos=graph.wire_positions[w])
        node_labels[w] = f"{wire}\n{graph.X_wires[w]}"
        
        # Color based on condition
        colors.append("red" if graph.X_wires[w][2] == 5.0 else "blue")

    print(f"Number of terminals: {len(graph.terminals)}")
    print(f"Number of terminal features: {len(graph.X_terminals)}")
    # Add terminal nodes
    wire_count = len(graph.detected_wires)  # Number of wire nodes
    for t, terminal in enumerate(graph.terminals):
        t_idx = wire_count + t  # Ensure terminal nodes have unique indices
        G.add_node(t_idx, pos=graph.terminal_positions[t])
        node_labels[t_idx] = f"Terminal_{terminal}\n{graph.X_terminals[t]}"  # Fixed indexing
        colors.append("blue")

    # Add edges
    edges = graph.get_edge_index().t().tolist()
    for src, tgt in edges:
        G.add_edge(src, tgt)

    # Compute layout
    pos = nx.kamada_kawai_layout(G)

    # Debugging: Check labels
    print(node_labels)
    
    plt.figure(figsize=(20, 17))
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=node_labels, node_size=4500, font_size=8, width=3.0)
    dir = f"../docs/figures/{year}_{month}_{day}/"
    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, f"{hour}{minute}.pdf")
    plt.savefig(save_path)
    
    plt.show()