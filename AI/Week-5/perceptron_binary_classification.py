"""
Perceptron Model for Binary Classification

This module demonstrates the implementation of a simple perceptron using TensorFlow/Keras
for binary classification tasks. Includes data generation, model training, evaluation,
and comprehensive visualization.

Features:
- Simple perceptron implementation
- Binary classification with synthetic data
- Real dataset support (breast cancer, make_classification)
- Model training and evaluation
- Decision boundary visualization
- Learning curve analysis
- Interactive demonstrations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PerceptronClassifier:
    """
    A comprehensive Perceptron classifier using TensorFlow/Keras
    """
    
    def __init__(self, learning_rate=0.01, activation='sigmoid'):
        """
        Initialize the Perceptron classifier
        
        Args:
            learning_rate: Learning rate for optimization
            activation: Activation function ('sigmoid', 'tanh')
        """
        self.learning_rate = learning_rate
        self.activation = activation
        self.model = None
        self.history = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_model(self, input_dim):
        """
        Create a simple perceptron model
        
        Args:
            input_dim: Number of input features
        """
        self.model = keras.Sequential([
            layers.Dense(1, 
                        input_dim=input_dim,
                        activation=self.activation,
                        name='perceptron_layer')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Perceptron Model Architecture:")
        self.model.summary()
        
    def prepare_data(self, X, y, test_size=0.2, scale_features=True):
        """
        Prepare data for training
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            scale_features: Whether to scale features
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        print(f"Data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Input features: {X_train.shape[1]}")
        print(f"  Features scaled: {scale_features}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the perceptron model
        
        Args:
            X_train: Training features
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
            verbose: Verbosity level
            
        Returns:
            History object containing training metrics
        """
        if self.model is None:
            self.create_model(X_train.shape[1])
        
        print(f"\nTraining Perceptron...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Activation: {self.activation}")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        self.is_fitted = True
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation!")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_prob.flatten(),
            'fpr': fpr,
            'tpr': tpr
        }
        
        return metrics
    
    def print_evaluation_results(self, y_test, metrics):
        """
        Print comprehensive evaluation results
        
        Args:
            y_test: True test labels
            metrics: Dictionary of evaluation metrics
        """
        print("\nPERCEPTRON EVALUATION RESULTS")
        print("=" * 35)
        
        print("Performance Metrics:")
        print(f"  Test Loss: {metrics['test_loss']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  ROC AUC Score: {metrics['roc_auc']:.4f}")
        
        # Model parameters
        weights = self.model.get_weights()[0].flatten()
        bias = self.model.get_weights()[1][0]
        
        print("\nLearned Parameters:")
        print(f"  Bias: {bias:.4f}")
        print("  Weights:")
        for i, weight in enumerate(weights):
            print(f"    Feature {i+1}: {weight:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, metrics['predictions']))
        
        # Confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, metrics['predictions'])
        print(cm)
    
    def visualize_training_history(self):
        """
        Visualize training history (loss and accuracy)
        """
        if self.history is None:
            raise ValueError("No training history available!")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
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
        fig.suptitle('Perceptron Binary Classification Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, metrics['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 2. ROC Curve
        plt.subplot(2, 2, 2)
        plt.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
                label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Prediction Distribution
        plt.subplot(2, 2, 3)
        class_0_probs = metrics['probabilities'][y_test == 0]
        class_1_probs = metrics['probabilities'][y_test == 1]
        
        plt.hist(class_0_probs, bins=20, alpha=0.7, label='Class 0', color='red')
        plt.hist(class_1_probs, bins=20, alpha=0.7, label='Class 1', color='blue')
        plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Boundary')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        
        # 4. Decision Boundary (for 2D data)
        plt.subplot(2, 2, 4)
        if X_test.shape[1] == 2:
            self._plot_decision_boundary_2d(X_test, y_test, metrics)
        else:
            # For higher dimensional data, show feature importance
            weights = self.model.get_weights()[0].flatten()
            feature_names = [f'Feature {i+1}' for i in range(len(weights))]
            
            plt.barh(feature_names, np.abs(weights))
            plt.xlabel('Absolute Weight Value')
            plt.title('Feature Importance (Absolute Weights)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_decision_boundary_2d(self, X_test, y_test, metrics):
        """
        Plot decision boundary for 2D data
        """
        # Create a mesh for plotting decision boundary
        h = 0.02  # step size in the mesh
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(mesh_points, verbose=0)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdGy')
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        # Plot data points
        scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                            cmap='RdGy', edgecolors='black')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.colorbar(scatter)

def create_synthetic_dataset(n_samples=1000, n_features=2, random_state=42):
    """
    Create synthetic binary classification dataset
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
        
    Returns:
        tuple: (X, y) features and targets
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    print(f"Synthetic Dataset Created:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y

def load_breast_cancer_dataset():
    """
    Load the breast cancer dataset for binary classification
    
    Returns:
        tuple: (X, y) features and targets
    """
    print("Loading Breast Cancer Dataset...")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(data.feature_names)}")
    print(f"  Classes: {data.target_names}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y

def demonstrate_perceptron_binary_classification():
    """
    Demonstrate perceptron for binary classification
    """
    print("PERCEPTRON BINARY CLASSIFICATION DEMONSTRATION")
    print("=" * 55)
    
    # Create synthetic 2D dataset for visualization
    X, y = create_synthetic_dataset(n_samples=500, n_features=2)
    
    # Initialize perceptron
    perceptron = PerceptronClassifier(learning_rate=0.1, activation='sigmoid')
    
    # Prepare data
    X_train, X_test, y_train, y_test = perceptron.prepare_data(X, y, scale_features=True)
    
    # Train the model
    history = perceptron.train(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    
    # Evaluate the model
    metrics = perceptron.evaluate(X_test, y_test)
    
    # Print results
    perceptron.print_evaluation_results(y_test, metrics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    perceptron.visualize_training_history()
    perceptron.visualize_results(X_test, y_test, metrics)
    
    return perceptron, metrics

def hyperparameter_comparison():
    """
    Compare different hyperparameter configurations
    """
    print("\nHYPERPARAMETER COMPARISON")
    print("=" * 30)
    
    # Load dataset
    X, y = create_synthetic_dataset(n_samples=1000, n_features=5)
    
    # Different configurations to test
    configs = [
        {'learning_rate': 0.01, 'activation': 'sigmoid', 'epochs': 50},
        {'learning_rate': 0.1, 'activation': 'sigmoid', 'epochs': 50},
        {'learning_rate': 0.01, 'activation': 'tanh', 'epochs': 50},
        {'learning_rate': 0.1, 'activation': 'tanh', 'epochs': 50},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}: {config}")
        
        # Initialize and train perceptron
        perceptron = PerceptronClassifier(
            learning_rate=config['learning_rate'],
            activation=config['activation']
        )
        
        X_train, X_test, y_train, y_test = perceptron.prepare_data(X, y)
        perceptron.train(X_train, y_train, epochs=config['epochs'], verbose=0)
        metrics = perceptron.evaluate(X_test, y_test)
        
        results.append({
            'Config': f"LR={config['learning_rate']}, Act={config['activation']}",
            'Test_Accuracy': metrics['test_accuracy'],
            'Test_Loss': metrics['test_loss'],
            'ROC_AUC': metrics['roc_auc']
        })
        
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Display comparison table
    print("\nCOMPARISON RESULTS:")
    print("=" * 60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

def interactive_perceptron_demo():
    """
    Interactive demonstration of perceptron classification
    """
    print("Welcome to Perceptron Binary Classification Demo!")
    print("=" * 55)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Basic Perceptron Demo (2D synthetic data)\n"
                          "2. Breast Cancer Dataset Demo\n"
                          "3. Hyperparameter Comparison\n"
                          "4. Custom Perceptron Training\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                demonstrate_perceptron_binary_classification()
                
            elif choice == '2':
                # Breast cancer dataset demo
                X, y = load_breast_cancer_dataset()
                
                perceptron = PerceptronClassifier(learning_rate=0.01, activation='sigmoid')
                X_train, X_test, y_train, y_test = perceptron.prepare_data(X, y)
                
                perceptron.train(X_train, y_train, epochs=100, verbose=1)
                metrics = perceptron.evaluate(X_test, y_test)
                perceptron.print_evaluation_results(y_test, metrics)
                
                show_viz = input("Show visualizations? (y/n): ").lower().strip()
                if show_viz == 'y':
                    perceptron.visualize_training_history()
                    perceptron.visualize_results(X_test, y_test, metrics)
                
            elif choice == '3':
                hyperparameter_comparison()
                
            elif choice == '4':
                # Custom training
                n_features = int(input("Enter number of features (2-20): ") or "5")
                learning_rate = float(input("Enter learning rate (0.001-1.0): ") or "0.01")
                activation = input("Enter activation (sigmoid/tanh): ") or "sigmoid"
                epochs = int(input("Enter number of epochs (10-200): ") or "50")
                
                X, y = create_synthetic_dataset(n_features=n_features)
                
                perceptron = PerceptronClassifier(
                    learning_rate=learning_rate,
                    activation=activation
                )
                
                X_train, X_test, y_train, y_test = perceptron.prepare_data(X, y)
                perceptron.train(X_train, y_train, epochs=epochs, verbose=1)
                metrics = perceptron.evaluate(X_test, y_test)
                
                perceptron.print_evaluation_results(y_test, metrics)
                perceptron.visualize_training_history()
                
            elif choice == '5':
                print("Thank you for using Perceptron Demo!")
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
    print("Perceptron Binary Classification - Quick Demo")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    try:
        print(f"Keras version: {tf.keras.__version__}")
    except AttributeError:
        print(f"Keras version: {tf.__version__} (integrated with TensorFlow)")
    
    # Run basic perceptron demo
    perceptron, metrics = demonstrate_perceptron_binary_classification()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_perceptron_demo()