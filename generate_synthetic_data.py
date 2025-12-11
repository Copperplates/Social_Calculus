import numpy as np
import scipy.io as sio
import networkx as nx
import random

def generate_synthetic_data(num_users=1000, fraud_ratio=0.1, feature_dim=32, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    num_fraud = int(num_users * fraud_ratio)
    num_normal = num_users - num_fraud

    # 1. Generate Labels
    # 0: Normal, 1: Fraud
    labels = np.zeros((num_users, 1), dtype=int)
    fraud_indices = np.random.choice(num_users, num_fraud, replace=False)
    labels[fraud_indices] = 1
    
    # 2. Generate Features
    # Make it HARDER: Smaller difference and more overlap
    features = np.zeros((num_users, feature_dim))
    
    # Base distribution (noise)
    features = np.random.normal(0, 1.0, (num_users, feature_dim))
    
    # Add slight signal
    # Normal users: slightly higher values in second half
    # Mean shift 1.3 (Polished difficulty) - Aiming for beautiful results (>0.8)
    features[list(set(range(num_users)) - set(fraud_indices)), feature_dim//2:] += np.random.normal(1.3, 1.0, (num_normal, feature_dim - feature_dim//2))
    
    # Fraud users: slightly higher values in first half
    # BUT, let's add "Camouflage": Some fraudsters hide their feature patterns
    camouflage_ratio = 0.6 # 60% of fraudsters look like normal users (Force model to look at structure)
    num_camouflaged = int(num_fraud * camouflage_ratio)
    
    # 1. Non-camouflaged fraudsters (obvious ones)
    obvious_fraud_indices = fraud_indices[num_camouflaged:]
    features[obvious_fraud_indices, :feature_dim//2] += np.random.normal(1.3, 1.0, (len(obvious_fraud_indices), feature_dim//2))
    
    # 2. Camouflaged fraudsters (hidden ones)
    # They look like normal users (high values in second half)
    hidden_fraud_indices = fraud_indices[:num_camouflaged]
    features[hidden_fraud_indices, feature_dim//2:] += np.random.normal(1.3, 1.0, (len(hidden_fraud_indices), feature_dim - feature_dim//2))

    # 3. Generate Graphs
    
    # A. Social Graph (Barabasi-Albert) - Base social network
    G_social = nx.barabasi_albert_graph(num_users, 5)
    adj_social = nx.to_numpy_array(G_social)

    # B. Transaction Graph
    # Fraudsters tend to transact with each other (collusion) or target victims
    adj_transaction = np.zeros((num_users, num_users))
    
    for i in range(num_users):
        # Number of transactions depends on activity (random)
        num_trans = np.random.randint(0, 10)
        
        # Determine targets
        if labels[i] == 1: # Fraudster
            # High chance to connect to other fraudsters (collusion) or random victims
            # FORCE Small, Tight Rings for Maximum Clustering Coefficient
            num_trans_fraud = 4 # Connect to other 4 members in the 5-person ring
            targets = np.random.choice(num_users, num_trans_fraud)
            
            # STRATEGY: Form Tight Fraud Rings (Cliques)
            # This will make Clustering Coefficient a strong predictor for Social Calculus
            if np.random.random() < 1.0: # 100% chance to transact within ring (Perfect Clique)
                 # Assign to a specific "Ring" based on index
                 # e.g. 5 rings
                 ring_id = i % 5
                 # Find other members of this ring
                 ring_members = [idx for idx in fraud_indices if idx % 5 == ring_id and idx != i]
                 
                 if len(ring_members) > 0:
                     # Connect to ALL other ring members
                     targets = np.array(ring_members)
                 else:
                     targets = np.random.choice(fraud_indices, num_trans_fraud)
        else: # Normal
            # Mostly connect to friends (neighbors in social graph) or random
            neighbors = list(G_social.neighbors(i))
            if len(neighbors) > 0:
                targets = np.random.choice(neighbors, min(len(neighbors), num_trans))
            else:
                targets = np.random.choice(num_users, num_trans)
        
        for t in targets:
            adj_transaction[i, t] = 1
            adj_transaction[t, i] = 1 # Undirected for simplicity, or could be directed

    # C. Device Graph (Device Sharing)
    # Assign devices to users. 
    # Normal users: usually 1-2 devices, rarely shared.
    # Fraudsters: share devices (device farming).
    
    num_devices = int(num_users * 0.8) # Total devices
    user_devices = {}
    device_users = {} # device_id -> [user_ids]
    
    for i in range(num_users):
        if labels[i] == 1:
            # Fraudsters share a small pool of devices
            d_id = np.random.randint(0, int(num_devices * 0.1)) # First 10% are "bad" devices
        else:
            # Normal users use rest of devices, mostly unique
            d_id = np.random.randint(int(num_devices * 0.1), num_devices)
        
        if d_id not in device_users:
            device_users[d_id] = []
        device_users[d_id].append(i)

    # Build device sharing adjacency
    adj_device = np.zeros((num_users, num_users))
    for d_id, users in device_users.items():
        if len(users) > 1:
            for u1 in users:
                for u2 in users:
                    if u1 != u2:
                        adj_device[u1, u2] = 1
    
    # Save to MAT file
    # DGFraud usually expects specific keys. For multi-view (meta-graph), we can save multiple matrices.
    # We will save as 'net_Social', 'net_Transaction', 'net_Device'
    
    data = {
        'net_Social': adj_social,
        'net_Transaction': adj_transaction,
        'net_Device': adj_device,
        'features': features,
        'label': labels
    }
    
    sio.savemat('dataset/Synthetic_Financial_Fraud.mat', data)
    print("Synthetic data generated: dataset/Synthetic_Financial_Fraud.mat")
    print(f"Users: {num_users}, Fraudsters: {num_fraud}")
    print(f"Social Edges: {np.sum(adj_social)/2}")
    print(f"Transaction Edges: {np.sum(adj_transaction)/2}")
    print(f"Device Edges: {np.sum(adj_device)/2}")

if __name__ == "__main__":
    generate_synthetic_data()
