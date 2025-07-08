# Week 3: Recurrent Neural Networks (RNNs)

## Overview
This week covers Recurrent Neural Networks including:
- Understanding RNNs by implementing from scratch
- Building RNNs with Keras for time series prediction and text generation
- Long Short-Term Memory (LSTM) networks for sequence modeling
- Gated Recurrent Units (GRU) and comparison with LSTMs
- Model evaluation, hyperparameter tuning, and visualization

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

### Task 1: Introduction to RNNs
**File**: `1_rnn_from_scratch.ipynb`
- Implement basic RNN cell from scratch using NumPy
- Understand the mathematical operations behind RNNs
- Forward and backward propagation through time
- Handle vanishing gradient problem

### Task 2: Building RNNs with Keras
**File**: `2_rnn_keras_timeseries.ipynb`
- Load and preprocess time series data
- Build RNN architecture with Keras
- Implement text generation using RNNs
- Train and evaluate models

### Task 3: Long Short-Term Memory (LSTM)
**File**: `3_lstm_implementation.ipynb`
- Implement LSTM for sequence modeling tasks
- Understand LSTM gates (forget, input, output)
- Compare LSTM performance with vanilla RNN
- Handle long-term dependencies

### Task 4: Gated Recurrent Units (GRU)
**File**: `4_gru_comparison.ipynb`
- Implement GRU networks
- Compare GRU vs LSTM performance
- Analyze computational efficiency
- Choose optimal architecture for specific tasks

### Task 5: Evaluation and Tuning
**File**: `5_evaluation_tuning.ipynb`
- Evaluate RNN models with appropriate metrics
- Hyperparameter tuning (learning rate, hidden units, layers)
- Visualize training progress and predictions
- Model interpretation and analysis

## Files Description

- `1_rnn_from_scratch.ipynb`: RNN implementation from scratch using NumPy
- `2_rnn_keras_timeseries.ipynb`: RNN with Keras for time series and text generation
- `3_lstm_implementation.ipynb`: LSTM implementation for sequence modeling
- `4_gru_comparison.ipynb`: GRU implementation and LSTM vs GRU comparison
- `5_evaluation_tuning.ipynb`: Model evaluation, tuning, and visualization
- `requirements.txt`: Required Python packages
- `README.md`: This documentation file

## Tasks Completed

✅ RNN implementation from scratch  
✅ RNN with Keras for time series prediction  
✅ LSTM implementation and sequence modeling  
✅ GRU implementation and performance comparison  
✅ Model evaluation, tuning, and visualization  

## Learning Outcomes

By the end of this week, you will:
- Understand the mathematical foundations of RNNs
- Be able to implement RNNs, LSTMs, and GRUs from scratch
- Know how to apply RNNs for time series prediction and text generation
- Understand the differences between RNN, LSTM, and GRU architectures
- Be able to evaluate and tune sequential models effectively
- Handle vanishing gradient problems in recurrent networks

## Datasets Used

- **Time Series**: Stock prices, weather data, or synthetic sequences
- **Text Generation**: Shakespeare, poetry, or custom text corpus
- **Sequence Classification**: IMDB reviews, sentiment analysis
- **Sequence-to-Sequence**: Machine translation, summarization

## Key Concepts Covered

- Recurrent connections and temporal dependencies
- Backpropagation through time (BPTT)
- Vanishing and exploding gradient problems
- LSTM gates: forget, input, cell, output
- GRU gates: reset, update
- Bidirectional RNNs
- Attention mechanisms (introduction)
- Sequence-to-sequence models

## Additional Considerations for This Week

### Best Practices for RNN Implementation

1. **Data Preprocessing**
   - Proper sequence normalization and scaling
   - Handling variable-length sequences with padding
   - Creating appropriate train/validation/test splits for time series

2. **Model Architecture Design**
   - Start with simpler models and gradually increase complexity
   - Use dropout for regularization
   - Consider bidirectional RNNs for non-causal relationships
   - Stack multiple RNN layers for complex patterns

3. **Training Optimization**
   - Use gradient clipping to prevent exploding gradients
   - Implement early stopping to prevent overfitting
   - Learning rate scheduling for better convergence
   - Batch size considerations for memory and stability

4. **Evaluation Strategies**
   - Use appropriate metrics for each task type
   - Implement cross-validation for time series data
   - Visualize predictions vs actual values
   - Analyze error patterns and residuals

### Common Pitfalls to Avoid

- **Data Leakage**: Ensure no future information in training data
- **Overfitting**: Monitor validation loss and use regularization
- **Scale Sensitivity**: Always normalize/standardize input data
- **Sequence Length**: Too short misses patterns, too long increases complexity
- **Batch Size**: Too small causes instability, too large reduces generalization

### Hyperparameter Tuning Guidelines

| Parameter | Range | Impact |
|-----------|-------|--------|
| Learning Rate | 1e-5 to 1e-1 | Convergence speed and stability |
| Hidden Units | 32 to 512 | Model capacity and overfitting |
| Sequence Length | 10 to 200 | Pattern capture vs computation |
| Dropout Rate | 0.1 to 0.5 | Regularization strength |
| Batch Size | 16 to 128 | Training stability and speed |

### Performance Comparison Framework

When comparing RNN variants, consider:
- **Accuracy**: Primary task-specific metrics
- **Training Time**: Computational efficiency
- **Memory Usage**: Resource requirements
- **Convergence**: Training stability and speed
- **Generalization**: Performance on unseen data

### Real-World Applications

1. **Time Series Forecasting**
   - Stock price prediction
   - Weather forecasting
   - Sales demand prediction

2. **Natural Language Processing**
   - Text generation
   - Sentiment analysis
   - Machine translation

3. **Sequential Pattern Recognition**
   - Anomaly detection
   - Music generation
   - Video analysis

### Advanced Topics for Further Exploration

- **Attention Mechanisms**: Improving long-sequence handling
- **Transformer Architecture**: State-of-the-art sequence modeling
- **Bidirectional RNNs**: Processing sequences in both directions
- **Encoder-Decoder Models**: Sequence-to-sequence tasks
- **Teacher Forcing**: Training strategy for sequence generation 