# 社会计算期末作业：基于图神经网络与社会信号增强的金融诈骗检测机制研究

这是我的社会计算课程期末作业。本项目使用图神经网络 (GNN) 分析社交平台上的金融欺诈行为，并模拟了不同的干预策略。

## 1. 环境配置

- **仓库**: `DGFraud-TF2` (TensorFlow 2.x 版本)
- **依赖**: TensorFlow >= 2.0, NumPy < 2.0 (修复了兼容性问题), SciPy, NetworkX.
- **已修复的问题**:
  - 在 `utils.py` 和主脚本中将 `np.bool` 替换为 `bool`。
  - 更新了 `layers/layers.py` 中的 `add_weight` 调用以兼容 Keras 3。
  - 更新了 `Adam` 优化器参数 (`lr` -> `learning_rate`)。
  - 修复了 `Player2Vec.py` 中的位置参数问题。

## 2. 数据生成

- **脚本**: `generate_synthetic_data.py`
- **输出**: `dataset/Synthetic_Financial_Fraud.mat`
- **内容**:
  - **节点**: 1000 个用户 (10% 为欺诈者)。
  - **图结构**:
    - `net_Social`: Barabasi-Albert 无标度网络。
    - `net_Transaction`: 交易网络 (欺诈者之间存在共谋)。
    - `net_Device`: 设备共享网络 (欺诈者共享设备)。
  - **特征**: 32 维向量 (欺诈者具有不同的分布)。

## 3. 欺诈检测模型

- **模型**: `Player2Vec` (关键玩家识别)。
- **脚本**: `run_synthetic_player2vec.py`
- **结果**: 在合成数据上实现了高准确率 (接近 100%)，证明了流水线的有效性以及特征具有区分度。

## 4. 干预模拟

- **脚本**: `simulate_intervention.py`
- **场景**:
  1. **无干预**: 欺诈者从 100 人扩散到约 286 人。
  2. **随机干预 (80% 检测率)**: 将受感染总人数减少至约 84 人。
  3. **策略干预 (Top 100 度中心性)**: 仅检查社交网络中度最高的前 100 个节点，虽然只移除了 30 个关键欺诈者，但将感染人数控制在约 103 人。
- **结论**: 针对社交枢纽 (高度节点) 进行干预是在资源有限的情况下遏制欺诈传播的高效策略。

## 如何运行

1. 生成数据:
   ```bash
   python generate_synthetic_data.py
   ```
2. 训练模型:
   ```bash
   python run_synthetic_player2vec.py
   ```
3. 模拟干预:
   ```bash
   python simulate_intervention.py
   ```
