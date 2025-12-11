
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm
import scipy.io as sio

from algorithms.Player2Vec.Player2Vec import Player2Vec
from utils.data_loader import load_data_synthetic
from utils.utils import preprocess_adj, preprocess_feature, sample_mask

# Set random seed
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

def train_and_evaluate(adj_list, features, label, masks, args, model_name="Baseline"):
    print(f"\n--- Training {model_name} Model ---")
    
    # Preprocess
    args.nodes = features.shape[0]
    train_mask, val_mask, test_mask = masks
    
    # Convert features to sparse tuple if needed (preprocess_feature does this)
    # But if we modify features, they might be dense.
    if sp.issparse(features):
        features_processed = preprocess_feature(features)
        features_tensor = tf.cast(tf.SparseTensor(*features_processed), dtype=tf.float32)
    else:
        # If dense (numpy array)
        features_sparse = sp.csr_matrix(features)
        features_processed = preprocess_feature(features_sparse)
        features_tensor = tf.cast(tf.SparseTensor(*features_processed), dtype=tf.float32)

    supports = [preprocess_adj(adj) for adj in adj_list]
    supports = [tf.cast(tf.SparseTensor(*support), dtype=tf.float32) for support in supports]

    # Model Config
    args.num_meta = len(supports)
    args.input_dim = features.shape[1]
    args.output_dim = label.shape[1]
    args.class_size = label.shape[1]
    # args.num_features_nonzero is needed for sparse dropout
    if sp.issparse(features):
        args.num_features_nonzero = features.nnz
    else:
        args.num_features_nonzero = np.count_nonzero(features)

    model = Player2Vec(args.input_dim, args.nhid, args.output_dim, args)
    optimizer = optimizers.Adam(learning_rate=args.lr)

    # Training Loop
    best_val_loss = float('inf')
    best_weights = None
    
    pbar = tqdm(range(args.epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        with tf.GradientTape() as tape:
            # Note: We pass training=True
            loss, acc = model([supports, features_tensor, label, train_mask], training=True)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Validation
        val_loss, val_acc = model([supports, features_tensor, label, val_mask], training=False)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            
        pbar.set_postfix({'loss': f"{loss:.4f}", 'val_loss': f"{val_loss:.4f}", 'val_acc': f"{val_acc:.4f}"})
    
    # Restore best weights
    model.set_weights(best_weights)
    
    # Final Evaluation (Test Set) with Embeddings
    test_loss, test_acc, logits, embeddings = model([supports, features_tensor, label, test_mask], training=False, return_embeddings=True)
    
    print(f"{model_name} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    return logits.numpy(), embeddings.numpy(), label.numpy(), test_mask.numpy()

def visualize_results(logits, embeddings, labels, mask, model_name):
    # Filter for test set
    test_indices = np.where(mask)[0]
    y_true = np.argmax(labels[test_indices], axis=1)
    y_scores = tf.nn.softmax(logits[test_indices]).numpy()[:, 1] # Prob of class 1 (Fraud)
    y_pred = np.argmax(logits[test_indices], axis=1)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    
    # 2. PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.subplot(2, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.subplot(2, 2, 3)
    # Simple heatmap
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.colorbar(im)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Fraud'])
    plt.yticks(tick_marks, ['Normal', 'Fraud'])
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # 4. t-SNE of Embeddings
    # Use test set embeddings
    test_embeddings = embeddings[test_indices]
    tsne = TSNE(n_components=2, random_state=SEED)
    emb_2d = tsne.fit_transform(test_embeddings)
    
    plt.subplot(2, 2, 4)
    colors = ['#4ECDC4' if y == 0 else '#FF6B6B' for y in y_true]
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.7, s=20)
    # Legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', label='Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', label='Fraud')
    ]
    plt.legend(handles=legend_elements)
    plt.title(f'{model_name} - t-SNE of Embeddings')
    
    plt.tight_layout()
    plt.savefig(f'analysis_results_{model_name}.png')
    plt.close()
    print(f"Saved visualization for {model_name} to analysis_results_{model_name}.png")

def analyze_network_characteristics(adj_list, labels_onehot):
    print("\n--- Analyzing Network Characteristics ---")
    labels = np.argmax(labels_onehot, axis=1)
    
    # Combine graphs
    adj_combined = sum(adj_list)
    adj_combined[adj_combined > 1] = 1
    G = nx.from_numpy_array(adj_combined)
    
    # 1. Homophily
    # Edge Homophily Ratio: Fraction of edges connecting nodes of same class
    same_class_edges = 0
    total_edges = 0
    for u, v in G.edges():
        total_edges += 1
        if labels[u] == labels[v]:
            same_class_edges += 1
    
    homophily = same_class_edges / total_edges if total_edges > 0 else 0
    print(f"Edge Homophily Ratio: {homophily:.4f}")
    
    # 2. Degree Distribution
    degrees = [d for n, d in G.degree()]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=30, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    
    # Log-Log Plot
    plt.subplot(1, 2, 2)
    # Count frequency
    from collections import Counter
    degree_counts = Counter(degrees)
    x = list(degree_counts.keys())
    y = list(degree_counts.values())
    plt.loglog(x, y, 'bo')
    plt.title("Degree Distribution (Log-Log)")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig('analysis_network_stats.png')
    plt.close()
    print("Saved network stats to analysis_network_stats.png")

def visualize_dataset(adj_list, features, labels_onehot):
    print("\n--- Visualizing Dataset ---")
    labels = np.argmax(labels_onehot, axis=1)
    
    # 1. Feature Space Visualization (PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    if sp.issparse(features):
        features_dense = features.toarray()
    else:
        features_dense = features
    
    X_pca = pca.fit_transform(features_dense)
    
    plt.figure(figsize=(15, 6))
    
    # Plot 1: PCA of Features
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[labels==0, 0], X_pca[labels==0, 1], c='#4ECDC4', label='Normal', alpha=0.6, s=20)
    plt.scatter(X_pca[labels==1, 0], X_pca[labels==1, 1], c='#FF6B6B', label='Fraud', alpha=0.6, s=20)
    plt.title('Feature Space (PCA)')
    plt.legend()
    
    # 2. Graph Structure Visualization (Transaction Graph - Subgraph)
    # We want to show the "Clique" structure of fraudsters
    plt.subplot(1, 2, 2)
    
    # Select a subset of nodes: All fraudsters + equal number of random normal users
    fraud_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]
    # Take first 20 fraudsters and 20 normal users for clarity
    selected_fraud = fraud_indices[:20]
    selected_normal = normal_indices[:20]
    selected_nodes = np.concatenate([selected_fraud, selected_normal])
    
    # Extract subgraph from Transaction Graph (Index 1)
    adj_trans = adj_list[1]
    # NetworkX subgraph
    # Need to handle sparse matrix if necessary, but usually adj_list are dense or sparse matrices
    if sp.issparse(adj_trans):
        adj_trans = adj_trans.toarray()
        
    G = nx.from_numpy_array(adj_trans)
    subgraph = G.subgraph(selected_nodes)
    
    # Layout
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
    
    # Draw
    node_colors = ['#FF6B6B' if node in fraud_indices else '#4ECDC4' for node in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color=node_colors)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3)
    
    plt.title('Transaction Network Structure (Subset)\nNote the Fraud Rings!')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('analysis_dataset_viz.png')
    plt.close()
    print("Saved dataset visualization to analysis_dataset_viz.png")

def visualize_comparison(results_baseline, results_enhanced):
    print("\n--- Visualizing Comparison ---")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Calculate metrics
    def get_metrics(y_true, y_pred):
        return [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred)
        ]
        
    base_metrics = get_metrics(results_baseline['y_true'], results_baseline['y_pred'])
    enh_metrics = get_metrics(results_enhanced['y_true'], results_enhanced['y_pred'])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, base_metrics, width, label='Baseline', color='#A0C4FF')
    plt.bar(x + width/2, enh_metrics, width, label='Enhanced (Social)', color='#FFADAD')
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(base_metrics):
        plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
    for i, v in enumerate(enh_metrics):
        plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig('analysis_comparison.png')
    plt.close()
    print("Saved comparison visualization to analysis_comparison.png")

def enhance_with_social_features(adj_list, features):
    print("\n--- Generating Social Features ---")
    # STRATEGY: Extract features specifically from Transaction Graph (Index 1)
    # Because Fraud Rings only exist in Transaction View!
    # Combined graph dilutes this signal with Social Graph noise.
    adj_target = adj_list[1] 
    
    # Ensure it's binary
    adj_target[adj_target > 1] = 1
    G = nx.from_numpy_array(adj_target)
    
    # 1. Degree Centrality
    deg_cent = nx.degree_centrality(G)
    
    # 2. PageRank
    pagerank = nx.pagerank(G)
    
    # 3. Clustering Coefficient
    clustering = nx.clustering(G)
    
    # 4. K-Core (might be slow for huge graphs, but fine for 1000 nodes)
    # core_number = nx.core_number(G) # Dictionary
    
    # Convert to arrays in order of node indices
    num_nodes = features.shape[0]
    social_feats = np.zeros((num_nodes, 3))
    
    for i in range(num_nodes):
        social_feats[i, 0] = deg_cent.get(i, 0)
        social_feats[i, 1] = pagerank.get(i, 0)
        social_feats[i, 2] = clustering.get(i, 0)
    
    # Normalize social features
    social_feats = (social_feats - np.mean(social_feats, axis=0)) / (np.std(social_feats, axis=0) + 1e-9)
    
    # Concatenate
    if sp.issparse(features):
        features_dense = features.toarray()
    else:
        features_dense = features
        
    enhanced_features = np.hstack((features_dense, social_feats))
    print(f"Original Feature Shape: {features.shape}")
    print(f"Enhanced Feature Shape: {enhanced_features.shape}")
    
    return enhanced_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_str', type=str, default='synthetic')
    parser.add_argument('--train_size', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=2) # Dummy, set later
    
    # Add dummy args expected by Player2Vec init
    parser.add_argument('--nodes', type=int, default=0)
    parser.add_argument('--class_size', type=int, default=0)
    parser.add_argument('--num_meta', type=int, default=0)
    parser.add_argument('--num_features_nonzero', type=int, default=0)
    
    args = parser.parse_args()
    
    # 1. Load Data
    adj_list, features, split_ids, y = load_data_synthetic(train_size=args.train_size)
    
    # Convert y to one-hot if needed
    if y.shape[1] == 1:
        y_flat = y.flatten().astype(int)
        n_values = np.max(y_flat) + 1
        y = np.eye(n_values)[y_flat]
    
    # Masks
    idx_train, _, idx_val, _, idx_test, _ = split_ids
    train_mask = tf.convert_to_tensor(sample_mask(idx_train, y.shape[0]))
    val_mask = tf.convert_to_tensor(sample_mask(idx_val, y.shape[0]))
    test_mask = tf.convert_to_tensor(sample_mask(idx_test, y.shape[0]))
    label = tf.convert_to_tensor(y, dtype=tf.float32)
    masks = [train_mask, val_mask, test_mask]
    
    # 2. Analyze Network
    analyze_network_characteristics(adj_list, y)
    visualize_dataset(adj_list, features, y)
    
    # 3. Train Baseline
    logits_base, embs_base, labels_base, mask_base = train_and_evaluate(adj_list, features, label, masks, args, "Baseline")
    visualize_results(logits_base, embs_base, labels_base, mask_base, "Baseline")
    
    # Capture results for comparison
    test_indices_base = np.where(mask_base)[0]
    results_baseline = {
        'y_true': np.argmax(labels_base[test_indices_base], axis=1),
        'y_pred': np.argmax(logits_base[test_indices_base], axis=1)
    }
    
    # 4. Train Enhanced (Social Signals)
    features_enhanced = enhance_with_social_features(adj_list, features)
    logits_enh, embs_enh, labels_enh, mask_enh = train_and_evaluate(adj_list, features_enhanced, label, masks, args, "Enhanced_Social")
    visualize_results(logits_enh, embs_enh, labels_enh, mask_enh, "Enhanced_Social")
    
    # Capture results for comparison
    test_indices_enh = np.where(mask_enh)[0]
    results_enhanced = {
        'y_true': np.argmax(labels_enh[test_indices_enh], axis=1),
        'y_pred': np.argmax(logits_enh[test_indices_enh], axis=1)
    }
    
    # 5. Compare
    visualize_comparison(results_baseline, results_enhanced)

if __name__ == "__main__":
    main()
