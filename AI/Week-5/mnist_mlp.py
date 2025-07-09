"""
Multi-Layer Perceptron (MLP) for MNIST Handwritten Digit Classification

This module demonstrates the implementation of a Multi-Layer Perceptron using
TensorFlow/Keras for the MNIST handwritten digit classification task. Includes
comprehensive training, evaluation, visualization, and hyperparameter tuning.

Features:
- MLP implementation with customizable architecture
- MNIST dataset loading and preprocessing
- Model training with visualization
- Comprehensive evaluation and metrics
- Hyperparameter tuning capabilities
- Prediction visualization and analysis
- Interactive demonstrations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MNISTMLPClassifier:
    """
    A comprehensive MLP classifier for MNIST digit classification
    """
    
    def __init__(self, hidden_layers=[128, 64], activation='relu', dropout_rate=0.2):
        """
        Initialize the MLP classifier
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        self.is_fitted = False
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the MNIST dataset
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) processed data
        """
        print("Loading MNIST Dataset...")
        print("=" * 25)
        
        # Load the MNIST dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print(f"Original data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape data to flatten images
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Convert labels to categorical (one-hot encoding)
        y_train_categorical = to_categorical(y_train, 10)
        y_test_categorical = to_categorical(y_test, 10)
        
        print(f"\nProcessed data shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train_categorical.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test_categorical.shape}")
        
        # Display class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass distribution in training set:")
        for digit, count in zip(unique, counts):
            print(f"  Digit {digit}: {count} samples")
        
        return X_train, X_test, y_train, y_test, y_train_categorical, y_test_categorical
    
    def visualize_sample_data(self, X_train, y_train, n_samples=10):
        """
        Visualize sample images from the dataset
        
        Args:
            X_train: Training images (flattened)
            y_train: Training labels
            n_samples: Number of samples to display
        """
        print(f"\nVisualizing {n_samples} sample images...")
        
        # Reshape back to 28x28 for visualization
        images = X_train[:n_samples].reshape(-1, 28, 28)
        labels = y_train[:n_samples]
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')
        
        for i, (image, label) in enumerate(zip(images, labels)):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'Digit: {label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_model(self, input_dim):
        """
        Create the MLP model architecture
        
        Args:
            input_dim: Number of input features (784 for MNIST)
        """
        print(f"\nCreating MLP Model...")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Activation: {self.activation}")
        print(f"  Dropout rate: {self.dropout_rate}")
        
        # Build the model
        model_layers = []
        
        # Input layer
        model_layers.append(layers.Dense(self.hidden_layers[0], 
                                       input_dim=input_dim, 
                                       activation=self.activation,
                                       name='hidden_1'))
        if self.dropout_rate > 0:
            model_layers.append(layers.Dropout(self.dropout_rate, name='dropout_1'))
        
        # Additional hidden layers
        for i, units in enumerate(self.hidden_layers[1:], 2):
            model_layers.append(layers.Dense(units, 
                                           activation=self.activation,
                                           name=f'hidden_{i}'))
            if self.dropout_rate > 0:
                model_layers.append(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))
        
        # Output layer (10 classes for digits 0-9)
        model_layers.append(layers.Dense(10, activation='softmax', name='output'))
        
        # Create the model
        self.model = keras.Sequential(model_layers)
        
        # Compile the model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
        
    def train(self, X_train, y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1):
        """
        Train the MLP model
        
        Args:
            X_train: Training features
            y_train: Training targets (categorical)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
            verbose: Verbosity level
            
        Returns:
            History object containing training metrics
        """
        if self.model is None:
            self.create_model(X_train.shape[1])
        
        print(f"\nTraining MLP Model...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation split: {validation_split}")
        
        # Add callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        return self.history
    
    def evaluate(self, X_test, y_test, y_test_categorical):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test targets (original labels)
            y_test_categorical: Test targets (categorical)
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation!")
        
        print("\nEvaluating Model...")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_categorical, verbose=0)
        
        # Per-class accuracy
        class_accuracies = []
        for digit in range(10):
            digit_mask = (y_test == digit)
            if np.sum(digit_mask) > 0:
                digit_accuracy = np.mean(y_pred[digit_mask] == y_test[digit_mask])
                class_accuracies.append(digit_accuracy)
            else:
                class_accuracies.append(0.0)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'class_accuracies': class_accuracies
        }
        
        return metrics
    
    def print_evaluation_results(self, y_test, metrics):
        """
        Print comprehensive evaluation results
        
        Args:
            y_test: True test labels
            metrics: Dictionary of evaluation metrics
        """
        print("\nMLP EVALUATION RESULTS")
        print("=" * 30)
        
        print("Overall Performance:")
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        # Per-digit accuracy
        print("\nPer-Digit Accuracy:")
        for digit, accuracy in enumerate(metrics['class_accuracies']):
            print(f"  Digit {digit}: {accuracy:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, metrics['predictions']))
    
    def visualize_training_history(self):
        """
        Visualize training history (loss and accuracy)
        """
        if self.history is None:
            raise ValueError("No training history available!")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('MLP Training History', fontsize=16, fontweight='bold')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_results(self, X_test, y_test, metrics):
        """
        Create comprehensive result visualizations
        
        Args:
            X_test: Test features
            y_test: Test targets
            metrics: Evaluation metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MNIST MLP Classification Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, metrics['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Digit')
        plt.ylabel('True Digit')
        
        # 2. Per-Class Accuracy
        plt.subplot(2, 2, 2)
        digits = range(10)
        plt.bar(digits, metrics['class_accuracies'], alpha=0.7)
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        plt.title('Per-Digit Classification Accuracy')
        plt.xticks(digits)
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for digit, acc in enumerate(metrics['class_accuracies']):
            plt.text(digit, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. Prediction Confidence Distribution
        plt.subplot(2, 2, 3)
        max_probs = np.max(metrics['probabilities'], axis=1)
        correct_mask = (metrics['predictions'] == y_test)
        
        plt.hist(max_probs[correct_mask], bins=30, alpha=0.7, 
                label='Correct Predictions', color='green')
        plt.hist(max_probs[~correct_mask], bins=30, alpha=0.7, 
                label='Incorrect Predictions', color='red')
        plt.xlabel('Maximum Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        
        # 4. Sample Predictions
        plt.subplot(2, 2, 4)
        # Show some sample predictions
        n_samples = 6
        sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 6, 7 + i)  # Start from row 2
            
            # Reshape and display image
            image = X_test[idx].reshape(28, 28)
            plt.imshow(image, cmap='gray')
            
            true_label = y_test[idx]
            pred_label = metrics['predictions'][idx]
            confidence = np.max(metrics['probabilities'][idx])
            
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f'T:{true_label} P:{pred_label}\n({confidence:.2f})', 
                     color=color, fontsize=8)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_misclassified_samples(self, X_test, y_test, metrics, n_samples=10):
        """
        Visualize misclassified samples for analysis
        
        Args:
            X_test: Test features
            y_test: Test targets
            metrics: Evaluation metrics
            n_samples: Number of misclassified samples to show
        """
        # Find misclassified samples
        misclassified_mask = (metrics['predictions'] != y_test)
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassified samples found!")
            return
        
        # Select random misclassified samples
        n_samples = min(n_samples, len(misclassified_indices))
        sample_indices = np.random.choice(misclassified_indices, n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        fig.suptitle('Misclassified Samples Analysis', fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(sample_indices):
            row = i // 5
            col = i % 5
            
            # Reshape and display image
            image = X_test[idx].reshape(28, 28)
            axes[row, col].imshow(image, cmap='gray')
            
            true_label = y_test[idx]
            pred_label = metrics['predictions'][idx]
            confidence = np.max(metrics['probabilities'][idx])
            
            axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Analyzed {n_samples} misclassified samples out of {len(misclassified_indices)} total misclassifications")

def demonstrate_mnist_mlp():
    """
    Demonstrate MLP classification on MNIST dataset
    """
    print("MNIST MLP CLASSIFICATION DEMONSTRATION")
    print("=" * 45)
    
    # Initialize MLP classifier
    mlp = MNISTMLPClassifier(hidden_layers=[128, 64], activation='relu', dropout_rate=0.2)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = mlp.load_and_preprocess_data()
    
    # Visualize sample data
    mlp.visualize_sample_data(X_train, y_train)
    
    # Train the model
    history = mlp.train(X_train, y_train_cat, epochs=20, batch_size=128, verbose=1)
    
    # Evaluate the model
    metrics = mlp.evaluate(X_test, y_test, y_test_cat)
    
    # Print results
    mlp.print_evaluation_results(y_test, metrics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    mlp.visualize_training_history()
    mlp.visualize_results(X_test, y_test, metrics)
    mlp.visualize_misclassified_samples(X_test, y_test, metrics)
    
    return mlp, metrics

def hyperparameter_tuning_demo():
    """
    Demonstrate hyperparameter tuning for MLP
    """
    print("\nHYPERPARAMETER TUNING DEMONSTRATION")
    print("=" * 40)
    
    # Load data (use subset for faster tuning)
    mlp = MNISTMLPClassifier()
    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = mlp.load_and_preprocess_data()
    
    # Use subset for faster experimentation
    subset_size = 10000
    X_train_subset = X_train[:subset_size]
    y_train_subset = y_train_cat[:subset_size]
    
    # Different configurations to test
    configs = [
        {'hidden_layers': [64], 'activation': 'relu', 'dropout_rate': 0.1, 'batch_size': 128},
        {'hidden_layers': [128, 64], 'activation': 'relu', 'dropout_rate': 0.2, 'batch_size': 128},
        {'hidden_layers': [256, 128], 'activation': 'relu', 'dropout_rate': 0.3, 'batch_size': 64},
        {'hidden_layers': [128, 64], 'activation': 'tanh', 'dropout_rate': 0.2, 'batch_size': 128},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        
        # Initialize and train MLP
        mlp_config = MNISTMLPClassifier(
            hidden_layers=config['hidden_layers'],
            activation=config['activation'],
            dropout_rate=config['dropout_rate']
        )
        
        mlp_config.train(X_train_subset, y_train_subset, 
                        epochs=10, batch_size=config['batch_size'], verbose=0)
        metrics = mlp_config.evaluate(X_test, y_test, y_test_cat)
        
        results.append({
            'Config': f"Layers={config['hidden_layers']}, Act={config['activation']}, Drop={config['dropout_rate']}",
            'Test_Accuracy': metrics['test_accuracy'],
            'Test_Loss': metrics['test_loss']
        })
        
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
    
    # Display comparison table
    print("\nHYPERPARAMETER TUNING RESULTS:")
    print("=" * 60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

def interactive_mnist_demo():
    """
    Interactive demonstration of MNIST MLP classification
    """
    print("Welcome to MNIST MLP Classification Demo!")
    print("=" * 45)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Basic MNIST MLP Demo\n"
                          "2. Hyperparameter Tuning Demo\n"
                          "3. Custom Architecture Training\n"
                          "4. Analyze Misclassifications\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                demonstrate_mnist_mlp()
                
            elif choice == '2':
                hyperparameter_tuning_demo()
                
            elif choice == '3':
                # Custom architecture
                print("Custom MLP Architecture:")
                
                # Get user input for architecture
                layers_input = input("Enter hidden layer sizes (e.g., '128,64,32'): ") or "128,64"
                hidden_layers = [int(x.strip()) for x in layers_input.split(',')]
                
                activation = input("Enter activation function (relu/tanh/sigmoid): ") or "relu"
                dropout_rate = float(input("Enter dropout rate (0.0-0.5): ") or "0.2")
                epochs = int(input("Enter number of epochs (5-50): ") or "15")
                batch_size = int(input("Enter batch size (32-256): ") or "128")
                
                # Train custom model
                mlp = MNISTMLPClassifier(
                    hidden_layers=hidden_layers,
                    activation=activation,
                    dropout_rate=dropout_rate
                )
                
                X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = mlp.load_and_preprocess_data()
                mlp.train(X_train, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=1)
                metrics = mlp.evaluate(X_test, y_test, y_test_cat)
                
                mlp.print_evaluation_results(y_test, metrics)
                
                show_viz = input("Show visualizations? (y/n): ").lower().strip()
                if show_viz == 'y':
                    mlp.visualize_training_history()
                    mlp.visualize_results(X_test, y_test, metrics)
                
            elif choice == '4':
                # Analyze misclassifications
                print("Training a quick model for misclassification analysis...")
                
                mlp = MNISTMLPClassifier()
                X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = mlp.load_and_preprocess_data()
                mlp.train(X_train[:10000], y_train_cat[:10000], epochs=10, verbose=0)
                metrics = mlp.evaluate(X_test, y_test, y_test_cat)
                
                mlp.visualize_misclassified_samples(X_test, y_test, metrics, n_samples=15)
                
            elif choice == '5':
                print("Thank you for using MNIST MLP Demo!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Quick demonstration
    print("MNIST MLP Classification - Quick Demo")
    print("=" * 40)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    try:
        print(f"Keras version: {tf.keras.__version__}")
    except AttributeError:
        print(f"Keras version: {tf.__version__} (integrated with TensorFlow)")
    
    # Run basic MNIST MLP demo
    mlp, metrics = demonstrate_mnist_mlp()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_mnist_demo() 