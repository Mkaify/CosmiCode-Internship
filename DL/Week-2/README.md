# Week 2: Convolutional Neural Networks (CNNs)

## Overview
This week covers Convolutional Neural Networks including:
- Understanding CNNs by implementing layers from scratch
- Building CNNs with Keras for image classification
- Data augmentation techniques
- Model optimization and hyperparameter tuning
- CNN visualization and interpretation

## Setup Instructions

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Tasks Description

### Task 1: Introduction to CNNs
**File**: `1_cnn_from_scratch.ipynb`
- Implement convolutional layer from scratch using NumPy
- Implement pooling layer from scratch
- Understand the mathematical operations behind CNNs

### Task 2: Building CNNs with Keras
**File**: `2_cnn_cifar10_keras.ipynb`
- Load and preprocess CIFAR-10 dataset
- Build CNN architecture with conv, pooling, and dense layers
- Train and evaluate the model

### Task 3: Data Augmentation
**File**: `3_data_augmentation.ipynb`
- Implement various augmentation techniques
- Compare model performance with and without augmentation
- Visualize augmented samples

### Task 4: Model Optimization
**File**: `4_model_optimization.ipynb`
- Experiment with different optimizers (SGD, Adam, RMSprop)
- Test various learning rates and regularization techniques
- Perform hyperparameter tuning

### Task 5: CNN Visualization
**File**: `5_cnn_visualization.ipynb`
- Visualize learned filters and feature maps
- Understand what CNNs learn at different layers
- Plot training history and performance metrics

## Files Description

- `1_cnn_from_scratch.ipynb`: CNN implementation from scratch using NumPy
- `2_cnn_cifar10_keras.ipynb`: Complete CNN for CIFAR-10 classification
- `3_data_augmentation.ipynb`: Data augmentation techniques and comparison
- `4_model_optimization.ipynb`: Hyperparameter tuning and optimization
- `5_cnn_visualization.ipynb`: CNN interpretation and visualization
- `requirements.txt`: Required Python packages
- `README.md`: This documentation file

## Tasks Completed

✅ CNN layers implementation from scratch  
✅ CIFAR-10 CNN with Keras  
✅ Data augmentation implementation  
✅ Model optimization and tuning  
✅ CNN visualization and interpretation  

## Learning Outcomes

By the end of this week, you will:
- Understand the mathematical foundations of CNNs
- Be able to build and train CNNs for image classification
- Know how to improve model performance with data augmentation
- Understand hyperparameter tuning for CNN optimization
- Be able to interpret and visualize CNN learned features 