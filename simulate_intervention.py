import numpy as np
import scipy.io as sio
import networkx as nx
import matplotlib.pyplot as plt

def load_graph(path='dataset/Synthetic_Financial_Fraud.mat'):
    data = sio.loadmat(path)
    # Combine social and transaction graphs for propagation
    adj_social = data['net_Social']
    adj_trans = data['net_Transaction']
    
    # Create a combined graph where edges are union of social and transaction
    # We assume fraud spreads through both: social influence and direct transactions
    adj_combined = adj_social + adj_trans
    adj_combined[adj_combined > 1] = 1
    
    G = nx.from_numpy_array(adj_combined)
    labels = data['label'].flatten()
    return G, labels

def simulate_propagation(G, initial_infected, p=0.1, max_steps=10):
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
            break
            
    return history

def run_intervention_experiment():
    G, labels = load_graph()
    num_users = len(labels)
    fraudsters = np.where(labels == 1)[0]
    
    print(f"Total Users: {num_users}")
    print(f"Initial Fraudsters: {len(fraudsters)}")
    
    # Scenario 1: No Intervention
    print("\n--- Scenario 1: No Intervention ---")
    history_baseline = simulate_propagation(G, fraudsters, p=0.05, max_steps=20)
    print(f"Infected count history: {history_baseline}")
    print(f"Final infected: {history_baseline[-1]}")
    
    # Scenario 2: Intervention (Block Detected Fraudsters)
    # Assume we detect 80% of fraudsters and block them (remove from graph)
    print("\n--- Scenario 2: Intervention (Block 80% Detected Fraudsters) ---")
    
    # Detection
    detected_fraudsters = np.random.choice(fraudsters, int(len(fraudsters) * 0.8), replace=False)
    
    # Intervention: Remove nodes
    G_intervened = G.copy()
    G_intervened.remove_nodes_from(detected_fraudsters)
    
    # Remaining fraudsters start the propagation
    remaining_fraudsters = [f for f in fraudsters if f not in detected_fraudsters]
    
    if len(remaining_fraudsters) > 0:
        history_intervention = simulate_propagation(G_intervened, remaining_fraudsters, p=0.05, max_steps=20)
        # Adjust count to include the removed (detected) ones as "neutralized" or just track active infected?
        # Usually we care about how many NORMAL users got infected.
        # Let's count total infected (including initial remaining fraudsters)
        print(f"Infected count history (active graph): {history_intervention}")
        print(f"Final infected (active graph): {history_intervention[-1]}")
    else:
        print("All fraudsters detected and blocked! No propagation.")

    # Scenario 3: Social Signal Based Intervention (Block High Degree Fraudsters first)
    # Assume we have limited resources, can only check top 100 high degree nodes
    print("\n--- Scenario 3: Strategic Intervention (Check Top 100 High Degree Nodes) ---")
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
    
    # Check if they are fraudsters
    detected_high_risk = [n for n in top_nodes if labels[n] == 1]
    print(f"Found {len(detected_high_risk)} fraudsters in top 100 high degree nodes.")
    
    G_strategic = G.copy()
    G_strategic.remove_nodes_from(detected_high_risk)
    
    remaining_fraudsters_strat = [f for f in fraudsters if f not in detected_high_risk]
    
    history_strategic = simulate_propagation(G_strategic, remaining_fraudsters_strat, p=0.05, max_steps=20)
    print(f"Infected count history: {history_strategic}")
    print(f"Final infected: {history_strategic[-1]}")

if __name__ == "__main__":
    run_intervention_experiment()
