import numpy as np
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt
import random

def load_graph_data(path='dataset/Synthetic_Financial_Fraud.mat'):
    data = sio.loadmat(path)
    adj_social = data['net_Social']
    adj_trans = data['net_Transaction']
    adj_device = data['net_Device']
    labels = data['label'].flatten()
    return adj_social, adj_trans, adj_device, labels

def plot_graph_structure(adj_social, adj_trans, labels):
    print("Generating Graph Structure Visualization...")
    # Combine graphs for visualization
    # We want to show a subgraph around a fraudster
    
    # Find a fraudster with some connections
    fraudster_indices = np.where(labels == 1)[0]
    
    # Build a temporary full graph to find a good subgraph
    # Using just Social + Transaction for this view
    adj_combined = adj_social + adj_trans
    G_full = nx.from_numpy_array(adj_combined)
    
    # Select a fraudster with reasonable degree
    selected_node = None
    for f_idx in fraudster_indices:
        if G_full.degree[f_idx] > 5 and G_full.degree[f_idx] < 20:
            selected_node = f_idx
            break
    
    if selected_node is None:
        selected_node = fraudster_indices[0]
        
    # Get 2-hop neighborhood
    subgraph_nodes = list(nx.single_source_shortest_path_length(G_full, selected_node, cutoff=2).keys())
    # Limit size if too big
    if len(subgraph_nodes) > 50:
        subgraph_nodes = subgraph_nodes[:50]
        
    G_sub = G_full.subgraph(subgraph_nodes)
    
    # Node Colors
    node_colors = []
    for node in G_sub.nodes():
        if labels[node] == 1:
            node_colors.append('#FF6B6B') # Red for Fraudster
        else:
            node_colors.append('#4ECDC4') # Teal for Normal
            
    # Layout
    pos = nx.spring_layout(G_sub, seed=42)
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=300, edgecolors='grey')
    nx.draw_networkx_edges(G_sub, pos, alpha=0.3)
    nx.draw_networkx_labels(G_sub, pos, font_size=8)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal User', markerfacecolor='#4ECDC4', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Fraudster', markerfacecolor='#FF6B6B', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title(f"Fraud Network Local Structure (Ego-net of Node {selected_node})")
    plt.axis('off')
    
    output_path = 'analysis_graph_structure.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved graph visualization to {output_path}")

def simulate_propagation(G, initial_infected, p=0.1, max_steps=20):
    infected = set(initial_infected)
    newly_infected = set(initial_infected)
    history = [len(infected)]
    
    for step in range(max_steps):
        next_newly_infected = set()
        for node in newly_infected:
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in infected:
                    if np.random.random() < p:
                        infected.add(neighbor)
                        next_newly_infected.add(neighbor)
        
        newly_infected = next_newly_infected
        history.append(len(infected))
        
        if len(newly_infected) == 0:
            # Fill remaining steps with last value
            history.extend([len(infected)] * (max_steps - step - 1))
            break
            
    return history

def plot_intervention_results(adj_social, adj_trans, labels):
    print("Generating Intervention Simulation Results...")
    
    # Prepare Graph
    adj_combined = adj_social + adj_trans
    adj_combined[adj_combined > 1] = 1
    G = nx.from_numpy_array(adj_combined)
    
    fraudsters = np.where(labels == 1)[0]
    
    # Parameters
    p_infect = 0.05
    steps = 20
    
    # 1. Baseline
    history_baseline = simulate_propagation(G, fraudsters, p=p_infect, max_steps=steps)
    
    # 2. Random Intervention (80% removal)
    detected_random = np.random.choice(fraudsters, int(len(fraudsters) * 0.8), replace=False)
    G_random = G.copy()
    G_random.remove_nodes_from(detected_random)
    remaining_random = [f for f in fraudsters if f not in detected_random]
    history_random = simulate_propagation(G_random, remaining_random, p=p_infect, max_steps=steps)
    
    # 3. Strategic Intervention (Top 100 Degree)
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
    detected_strategic = [n for n in top_nodes if labels[n] == 1]
    
    G_strategic = G.copy()
    G_strategic.remove_nodes_from(detected_strategic)
    remaining_strategic = [f for f in fraudsters if f not in detected_strategic]
    history_strategic = simulate_propagation(G_strategic, remaining_strategic, p=p_infect, max_steps=steps)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    x = range(len(history_baseline))
    
    plt.plot(x, history_baseline, 'r-o', linewidth=2, label='No Intervention')
    plt.plot(x, history_random, 'b--s', linewidth=2, label='Random Intervention (Block 80% Fraudsters)')
    plt.plot(x, history_strategic, 'g-.^', linewidth=2, label='Strategic Intervention (Block Top-Degree Nodes)')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Total Infected Nodes')
    plt.title('Impact of Intervention Strategies on Fraud Propagation')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_path = 'analysis_intervention_results.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved intervention results to {output_path}")

if __name__ == "__main__":
    adj_social, adj_trans, adj_device, labels = load_graph_data()
    plot_graph_structure(adj_social, adj_trans, labels)
    plot_intervention_results(adj_social, adj_trans, labels)
