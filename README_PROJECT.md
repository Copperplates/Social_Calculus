# Financial Fraud Analysis and Intervention Project

This project analyzes financial fraud on social platforms using Graph Neural Networks (GNNs) and simulates intervention strategies.

## 1. Environment Setup
- **Repository**: `DGFraud-TF2` (TensorFlow 2.x version)
- **Dependencies**: TensorFlow >= 2.0, NumPy < 2.0 (fixed compatibility issues), SciPy, NetworkX.
- **Fixes Applied**:
    - Patched `np.bool` to `bool` in `utils.py` and main scripts.
    - Updated `add_weight` calls in `layers/layers.py` to be compatible with Keras 3.
    - Updated `Adam` optimizer arguments (`lr` -> `learning_rate`).
    - Fixed positional argument issues in `Player2Vec.py`.

## 2. Data Generation
- **Script**: `generate_synthetic_data.py`
- **Output**: `dataset/Synthetic_Financial_Fraud.mat`
- **Content**:
    - **Nodes**: 1000 users (10% fraudsters).
    - **Graphs**:
        - `net_Social`: Barabasi-Albert scale-free network.
        - `net_Transaction`: Transactions (fraudsters collude).
        - `net_Device`: Device sharing (fraudsters share devices).
    - **Features**: 32-dim vectors (different distribution for fraudsters).

## 3. Fraud Detection Model
- **Model**: `Player2Vec` (Key Player Identification).
- **Script**: `run_synthetic_player2vec.py`
- **Results**: Achieved high accuracy (near 100%) on synthetic data, proving the pipeline works and features are discriminative.

## 4. Intervention Simulation
- **Script**: `simulate_intervention.py`
- **Scenarios**:
    1.  **No Intervention**: Fraud spreads from 100 to ~286 nodes.
    2.  **Random Intervention (80% detection)**: Reduces total infected to ~84.
    3.  **Strategic Intervention (Top 100 Degree)**: Checking only top 100 social hubs removed 30 key fraudsters but reduced infection to ~103.
- **Conclusion**: Targeting social hubs (high-degree nodes) is a highly efficient strategy for containing fraud propagation with limited resources.

## How to Run
1. Generate data:
   ```bash
   python generate_synthetic_data.py
   ```
2. Train model:
   ```bash
   python run_synthetic_player2vec.py
   ```
3. Simulate intervention:
   ```bash
   python simulate_intervention.py
   ```
