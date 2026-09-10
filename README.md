## Despoina Christina Markatou - Diploma Thesis, NTUA

# Efficient AI Workload Orchestration in Edge-Cloud Infrastructure using DRL

## 📖 Overview
This repository contains the implementation of the Diploma Thesis: **"Efficient Artificial Intelligence Workload Orchestration in Edge-Cloud Infrastructure using Deep Reinforcement Learning"**. 

As AI models grow in complexity, executing both training and inference workloads requires distributed Edge-Cloud infrastructures. However, orchestrating these resources is a highly complex multi-objective optimization problem. This project proposes an intelligent orchestration framework driven by a **Deep Q-Network (DQN) agent** that continuously monitors system telemetry to find the optimal placement and parallelization strategy for each incoming AI workload. 

The primary optimization objectives are the simultaneous minimization of **execution cost** and **resource idle time**.

## Key Technical Features
* **Deep Reinforcement Learning (DQN) Agent:** A DQN agent utilizing an Experience Replay Buffer and a Dual Network architecture to ensure training stability.
* **Action Space Top-K Masking:** A preprocessing module that filters invalid execution profiles based on strict hardware and SLA constraints and masks the action space to a fixed $K$ dimension.
* **One-Step-Ahead Heuristic:** A model-based approach utilized during the inference phase that pre-computes immediate rewards to reduce the neural network's estimation errors.
* **Multi-Objective Reward Function:** A Min-Max normalized reward mechanism that perfectly balances economic execution costs with hardware idle time, while enforcing strict penalties for task rejections.
* **Algorithmic Baselines:** Includes implementations of a fast Greedy Heuristic and a theoretical MILP (Mixed Integer Linear Programming) solver for comprehensive benchmarking.

## Supported Parallelization Strategies
The framework dynamically calculates computation and communication delays to select the optimal distributed execution strategy based on hardware constraints and network topology:
* **Data Parallelism:** Supports both centralized **Parameter Server** architectures and decentralized **Ring All-Reduce** for efficient gradient synchronization across multiple edge/cloud nodes.
* **Model Parallelism:** Implements **Tensor Parallelism** (horizontal intra-layer partitioning for concurrent matrix multiplications) and **Pipeline Parallelism** (vertical sequential partitioning utilizing micro-batching) to handle massive models that exceed the VRAM capacity of a single GPU.
* **Single Node Execution:** Automatically selected for lightweight inference tasks or small-scale local training to eliminate network overhead.

## 📂 Repository Structure

```text
├── src/                      # Core system implementation
│   ├── environment.py        # Edge-Cloud infrastructure simulator
│   ├── dqn_agent.py          # Deep Q-Network agent architecture
│   └── preprocessing.py      # Profile generation and Top-K masking
├── data/                     # Synthetically generated workload JSONs
│   ├── small
|   |  ├──dataset_1_abundance.json
|   |  ├──dataset_1_abundance.json
|   |  ├──dataset_1_abundance.json
|   |  └──dataset_1_abundance.json
│   └── big
|      ├──dataset_1_abundance.json
|      ├──dataset_1_abundance.json
|      ├──dataset_1_abundance.json
|      └──dataset_1_abundance.json
├── milp/
|   └── milp.py               # MILP global optimal mathematical model
├── scripts/                  # Execution scripts separated by phase
│   ├── data_generation/      # Scripts to generate Poisson-distributed traffic
│   ├── training/             # DQN training, reward ablation, pruning training
│   ├── evaluation/           # Inference scripts for DQN, Greedy, and MILP
│   └── visualization/        # Python scripts for plotting metrics (matplotlib)
└── results/                  # Generated plots and metric evaluation grids
```

## 🚀 Getting Started

### Prerequisites
* Python 3.13.0
* PyTorch 2.12.0
* PuLP (for MILP evaluation)

### Installation
```bash
git clone https://github.com/cdemrk/thesis.git
cd thesis
python -m venv myenv
source myenv/bin/activate
```

### Running the Project
**1. Train the DRL Agent:**
```bash
python scripts/training/train_agent.py
```
**2. Evaluate under Heavy Contention:**
```bash
python scripts/evaluation/eval_drl.py --dataset data/dataset_4_heavy_contention.json
```
**3. Generate Performance Plots:**
```bash
python scripts/visualization/comparison_diagrams.py
```

## 📊 Key Findings
Extensive simulations proved that under resource abundance, a Greedy heuristic executes efficiently. However, as the infrastructure transitions into mild and heavy contention, the proposed DRL agent outperforms the heuristic by preventing premature resource exhaustion. The DRL agent achieves up to **94.1% of the theoretical MILP optimal service rate** while maintaining an ultra-low decision latency of `< 1.6 ms` per task via $K=4$ action masking. 

## 👨‍💻 Author
**Despoina Christina Markatou**  
*National Technical University of Athens (NTUA)*  
*School of Electrical and Computer Engineering*
