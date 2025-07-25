# Week 5: Specialized Deep Learning Techniques

## Overview
Advanced deep learning implementations covering object detection, semantic segmentation, reinforcement learning, graph neural networks, and hyperparameter optimization.

## Setup
```bash
pip install -r requirements.txt
jupyter notebook
```

## Implemented Tasks

### 1. Object Detection (`1_object_dectection.ipynb`)
- YOLO implementation with YOLOv5
- Real-time object detection
- COCO dataset integration
- mAP evaluation metrics

### 2. Semantic Segmentation (`2_semantic_segmentation.ipynb`)
- U-Net architecture for medical imaging
- Pixel-wise classification
- IoU metrics and evaluation
- Custom dataset handling

### 3. Reinforcement Learning (`3_reinforcement_learning.ipynb`)
- Deep Q-Network (DQN) implementation
- OpenAI Gym environments
- Experience replay and target networks
- Game playing agents

### 4. Graph Neural Networks (`4_graph_neural_networks.py`)
- Graph Convolutional Networks (GCNs)
- Node classification on citation networks
- PyTorch Geometric implementation
- Cora dataset processing

### 5. Hyperparameter Optimization (`5_hyperparameter_optimization.ipynb`)
- Keras Tuner implementation
- Bayesian optimization
- Grid and random search
- Automated model tuning

## Key Dependencies
- TensorFlow/Keras for deep learning
- PyTorch + PyTorch Geometric for GNNs
- OpenCV for computer vision
- Gym for reinforcement learning
- Optuna/Keras Tuner for optimization

## Results
- ✅ Working object detection with YOLO
- ✅ Semantic segmentation with U-Net
- ✅ RL agents trained on various environments  
- ✅ GNN models for graph classification
- ✅ Optimized hyperparameters using automated search

## Files Structure
- `1_object_dectection.ipynb` - YOLO object detection
- `2_semantic_segmentation.ipynb` - U-Net segmentation
- `3_reinforcement_learning.ipynb` - DQN implementation
- `4_graph_neural_networks.py` - GCN models
- `5_hyperparameter_optimization.ipynb` - Automated tuning
- `yolov5/` - YOLO model files
- `kt_dir/`, `seg_kt_dir/` - Keras Tuner results
- `requirements.txt` - Python dependencies

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Open Jupyter: `jupyter notebook`
3. Run notebooks in order or individually
4. Check result directories for trained models and outputs 