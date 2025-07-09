"""
Hyperparameter Tuning for Neural Networks

This module provides comprehensive tools for hyperparameter tuning of neural networks,
including systematic exploration of epochs, batch sizes, activation functions,
learning rates, and network architectures.

Features:
- Systematic hyperparameter exploration
- Grid search and random search capabilities
- Performance comparison and visualization
- Best practices for hyperparameter tuning
- Interactive tuning demonstrations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class HyperparameterTuner:
    """
    A comprehensive hyperparameter tuning class for neural networks
    """
    
    def __init__(self, task_type='classification'):
        """
        Initialize the hyperparameter tuner
        
        Args:
            task_type: Type of task ('classification' or 'regression')
        """
        self.task_type = task_type
        self.results = []
        self.best_model = None
        self.best_params = None
        self.best_score = 0 if task_type == 'classification' else float('inf')
        
    def create_model(self, input_dim, output_dim, hidden_layers, activation, learning_rate, dropout_rate):
        """
        Create a neural network model with specified hyperparameters
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes/units
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            learning_rate: Learning rate for optimizer
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()
        
        # First hidden layer
        model.add(layers.Dense(hidden_layers[0], 
                              input_dim=input_dim, 
                              activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
        
        # Additional hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation=activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if self.task_type == 'classification':
            if output_dim == 1:
                model.add(layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
            else:
                model.add(layers.Dense(output_dim, activation='softmax'))
                loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(output_dim, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        return model
    
    def evaluate_hyperparameters(self, X_train, y_train, X_val, y_val, hyperparams):
        """
        Evaluate a specific set of hyperparameters
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            hyperparams: Dictionary of hyperparameters
            
        Returns:
            Dictionary with results
        """
        print(f"Testing: {hyperparams}")
        
        # Create model
        model = self.create_model(
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1] if len(y_train.shape) > 1 else 1,
            hidden_layers=hyperparams['hidden_layers'],
            activation=hyperparams['activation'],
            learning_rate=hyperparams['learning_rate'],
            dropout_rate=hyperparams['dropout_rate']
        )
        
        # Train model
        start_time = time.time()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate model
        val_loss, val_metric = model.evaluate(X_val, y_val, verbose=0)
        
        # Calculate final scores from history
        final_train_loss = history.history['loss'][-1]
        final_train_metric = history.history[model.metrics_names[1]][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_metric = history.history[f'val_{model.metrics_names[1]}'][-1]
        
        # Check if this is the best model
        current_score = final_val_metric if self.task_type == 'classification' else -final_val_loss
        
        if ((self.task_type == 'classification' and current_score > self.best_score) or
            (self.task_type == 'regression' and current_score > self.best_score)):
            self.best_score = current_score
            self.best_model = model
            self.best_params = hyperparams.copy()
        
        result = {
            'hyperparams': hyperparams.copy(),
            'train_loss': final_train_loss,
            'train_metric': final_train_metric,
            'val_loss': final_val_loss,
            'val_metric': final_val_metric,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'overfitting': final_train_metric - final_val_metric if self.task_type == 'classification' else final_val_loss - final_train_loss
        }
        
        self.results.append(result)
        
        print(f"  Val {model.metrics_names[1]}: {final_val_metric:.4f}, Time: {training_time:.2f}s")
        
        return result
    
    def grid_search(self, X_train, y_train, X_val, y_val, param_grid):
        """
        Perform grid search over hyperparameters
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            param_grid: Dictionary of parameter lists to search
        """
        print("HYPERPARAMETER GRID SEARCH")
        print("=" * 35)
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"Testing {len(combinations)} hyperparameter combinations...")
        print(f"Parameters: {param_names}")
        print("-" * 50)
        
        # Test each combination
        for i, combination in enumerate(combinations, 1):
            hyperparams = dict(zip(param_names, combination))
            print(f"\nCombination {i}/{len(combinations)}:")
            
            try:
                self.evaluate_hyperparameters(X_train, y_train, X_val, y_val, hyperparams)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\nGrid search completed!")
        print(f"Best {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
    
    def random_search(self, X_train, y_train, X_val, y_val, param_distributions, n_trials=20):
        """
        Perform random search over hyperparameters
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            param_distributions: Dictionary of parameter distributions
            n_trials: Number of random trials
        """
        print("HYPERPARAMETER RANDOM SEARCH")
        print("=" * 35)
        
        print(f"Testing {n_trials} random hyperparameter combinations...")
        print("-" * 50)
        
        for trial in range(n_trials):
            # Sample random hyperparameters
            hyperparams = {}
            for param, values in param_distributions.items():
                if isinstance(values, list):
                    hyperparams[param] = np.random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # Assume (min, max) range
                    if isinstance(values[0], int):
                        hyperparams[param] = np.random.randint(values[0], values[1] + 1)
                    else:
                        hyperparams[param] = np.random.uniform(values[0], values[1])
                else:
                    hyperparams[param] = values
            
            print(f"\nTrial {trial + 1}/{n_trials}:")
            
            try:
                self.evaluate_hyperparameters(X_train, y_train, X_val, y_val, hyperparams)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\nRandom search completed!")
        print(f"Best {self.best_params}")
        print(f"Best score: {self.best_score:.4f}")
    
    def analyze_results(self):
        """
        Analyze and visualize hyperparameter tuning results
        """
        if not self.results:
            print("No results to analyze!")
            return
        
        # Convert results to DataFrame
        df_results = pd.DataFrame([
            {**result['hyperparams'], **{k: v for k, v in result.items() if k != 'hyperparams'}}
            for result in self.results
        ])
        
        print("\nHYPERPARAMETER TUNING ANALYSIS")
        print("=" * 40)
        
        # Best results
        metric_name = 'val_metric'
        if self.task_type == 'classification':
            best_idx = df_results[metric_name].idxmax()
        else:
            best_idx = df_results['val_loss'].idxmin()
        
        print("Top 5 Results:")
        print("-" * 20)
        
        sorted_df = df_results.sort_values(metric_name, ascending=False) if self.task_type == 'classification' else df_results.sort_values('val_loss')
        for i, (_, row) in enumerate(sorted_df.head().iterrows(), 1):
            print(f"{i}. Val Score: {row[metric_name]:.4f}, "
                  f"Layers: {row['hidden_layers']}, "
                  f"LR: {row['learning_rate']}, "
                  f"Activation: {row['activation']}")
        
        # Create visualizations
        self.visualize_hyperparameter_effects(df_results)
        
        return df_results
    
    def visualize_hyperparameter_effects(self, df_results):
        """
        Visualize the effects of different hyperparameters
        
        Args:
            df_results: DataFrame with tuning results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Effects Analysis', fontsize=16, fontweight='bold')
        
        metric_col = 'val_metric'
        
        # 1. Learning Rate Effect
        plt.subplot(2, 3, 1)
        lr_groups = df_results.groupby('learning_rate')[metric_col].agg(['mean', 'std']).reset_index()
        plt.errorbar(lr_groups['learning_rate'], lr_groups['mean'], yerr=lr_groups['std'], 
                    marker='o', capsize=5)
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Score')
        plt.title('Learning Rate Effect')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # 2. Batch Size Effect
        plt.subplot(2, 3, 2)
        batch_groups = df_results.groupby('batch_size')[metric_col].agg(['mean', 'std']).reset_index()
        plt.errorbar(batch_groups['batch_size'], batch_groups['mean'], yerr=batch_groups['std'], 
                    marker='o', capsize=5)
        plt.xlabel('Batch Size')
        plt.ylabel('Validation Score')
        plt.title('Batch Size Effect')
        plt.grid(True, alpha=0.3)
        
        # 3. Activation Function Effect
        plt.subplot(2, 3, 3)
        activation_stats = df_results.groupby('activation')[metric_col].agg(['mean', 'std'])
        plt.bar(activation_stats.index, activation_stats['mean'], yerr=activation_stats['std'],
               capsize=5, alpha=0.7)
        plt.xlabel('Activation Function')
        plt.ylabel('Validation Score')
        plt.title('Activation Function Effect')
        plt.xticks(rotation=45)
        
        # 4. Dropout Rate Effect
        plt.subplot(2, 3, 4)
        dropout_groups = df_results.groupby('dropout_rate')[metric_col].agg(['mean', 'std']).reset_index()
        plt.errorbar(dropout_groups['dropout_rate'], dropout_groups['mean'], yerr=dropout_groups['std'], 
                    marker='o', capsize=5)
        plt.xlabel('Dropout Rate')
        plt.ylabel('Validation Score')
        plt.title('Dropout Rate Effect')
        plt.grid(True, alpha=0.3)
        
        # 5. Training Time vs Performance
        plt.subplot(2, 3, 5)
        plt.scatter(df_results['training_time'], df_results[metric_col], alpha=0.6)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Validation Score')
        plt.title('Training Time vs Performance')
        plt.grid(True, alpha=0.3)
        
        # 6. Overfitting Analysis
        plt.subplot(2, 3, 6)
        plt.scatter(df_results['overfitting'], df_results[metric_col], alpha=0.6)
        plt.xlabel('Overfitting (Train - Val)')
        plt.ylabel('Validation Score')
        plt.title('Overfitting vs Performance')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def load_sample_dataset(dataset_type='digits'):
    """
    Load a sample dataset for hyperparameter tuning
    
    Args:
        dataset_type: Type of dataset ('digits', 'synthetic')
        
    Returns:
        Preprocessed training and validation data
    """
    if dataset_type == 'digits':
        # Load digits dataset (8x8 = 64 features, 10 classes)
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.data, digits.target
        
        print(f"Loaded digits dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
    elif dataset_type == 'synthetic':
        # Create synthetic classification dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=3,
            random_state=42
        )
        
        print(f"Created synthetic dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    
    # Preprocess data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to categorical for multi-class classification
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    
    return X_train, X_val, y_train_cat, y_val_cat

def demonstrate_grid_search():
    """
    Demonstrate grid search hyperparameter tuning
    """
    print("GRID SEARCH DEMONSTRATION")
    print("=" * 30)
    
    # Load data
    X_train, X_val, y_train, y_val = load_sample_dataset('digits')
    
    # Initialize tuner
    tuner = HyperparameterTuner(task_type='classification')
    
    # Define parameter grid (smaller for demonstration)
    param_grid = {
        'hidden_layers': [[64], [128], [64, 32]],
        'activation': ['relu', 'tanh'],
        'learning_rate': [0.001, 0.01],
        'dropout_rate': [0.1, 0.3],
        'batch_size': [32, 64],
        'epochs': [20]  # Fixed for faster execution
    }
    
    # Perform grid search
    tuner.grid_search(X_train, y_train, X_val, y_val, param_grid)
    
    # Analyze results
    df_results = tuner.analyze_results()
    
    return tuner, df_results

def demonstrate_random_search():
    """
    Demonstrate random search hyperparameter tuning
    """
    print("\nRANDOM SEARCH DEMONSTRATION")
    print("=" * 35)
    
    # Load data
    X_train, X_val, y_train, y_val = load_sample_dataset('synthetic')
    
    # Initialize tuner
    tuner = HyperparameterTuner(task_type='classification')
    
    # Define parameter distributions
    param_distributions = {
        'hidden_layers': [[32], [64], [128], [64, 32], [128, 64], [256, 128]],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'learning_rate': (0.0001, 0.1),  # Range for uniform sampling
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4],
        'batch_size': [16, 32, 64, 128],
        'epochs': [15]  # Fixed for faster execution
    }
    
    # Perform random search
    tuner.random_search(X_train, y_train, X_val, y_val, param_distributions, n_trials=15)
    
    # Analyze results
    df_results = tuner.analyze_results()
    
    return tuner, df_results

def compare_tuning_strategies():
    """
    Compare different hyperparameter tuning strategies
    """
    print("\nTUNING STRATEGY COMPARISON")
    print("=" * 35)
    
    # This would compare grid search, random search, and Bayesian optimization
    # For now, we'll show the concept with grid vs random
    
    print("Strategy Comparison Summary:")
    print("1. Grid Search:")
    print("   - Pros: Exhaustive, reproducible")
    print("   - Cons: Exponentially expensive, may miss optimal combinations")
    
    print("\n2. Random Search:")
    print("   - Pros: More efficient, can find good solutions faster")
    print("   - Cons: No guarantee of finding optimal solution")
    
    print("\n3. Bayesian Optimization (Advanced):")
    print("   - Pros: Intelligent search, learns from previous evaluations")
    print("   - Cons: More complex to implement, requires additional libraries")

def interactive_hyperparameter_demo():
    """
    Interactive demonstration of hyperparameter tuning
    """
    print("Welcome to Hyperparameter Tuning Demo!")
    print("=" * 45)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Grid Search Demo\n"
                          "2. Random Search Demo\n"
                          "3. Compare Tuning Strategies\n"
                          "4. Custom Hyperparameter Exploration\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                demonstrate_grid_search()
                
            elif choice == '2':
                demonstrate_random_search()
                
            elif choice == '3':
                compare_tuning_strategies()
                
            elif choice == '4':
                # Custom exploration
                print("Custom Hyperparameter Exploration:")
                
                dataset = input("Choose dataset (digits/synthetic): ") or "digits"
                search_type = input("Choose search type (grid/random): ") or "grid"
                
                X_train, X_val, y_train, y_val = load_sample_dataset(dataset)
                tuner = HyperparameterTuner(task_type='classification')
                
                if search_type == 'grid':
                    # Simple grid for custom exploration
                    param_grid = {
                        'hidden_layers': [[64], [128]],
                        'activation': ['relu', 'tanh'],
                        'learning_rate': [0.01, 0.001],
                        'dropout_rate': [0.2],
                        'batch_size': [32, 64],
                        'epochs': [15]
                    }
                    tuner.grid_search(X_train, y_train, X_val, y_val, param_grid)
                else:
                    # Random search
                    param_distributions = {
                        'hidden_layers': [[32], [64], [128]],
                        'activation': ['relu', 'tanh'],
                        'learning_rate': (0.001, 0.01),
                        'dropout_rate': [0.1, 0.2, 0.3],
                        'batch_size': [32, 64],
                        'epochs': [15]
                    }
                    tuner.random_search(X_train, y_train, X_val, y_val, param_distributions, n_trials=10)
                
                tuner.analyze_results()
                
            elif choice == '5':
                print("Thank you for using Hyperparameter Tuning Demo!")
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
    print("Hyperparameter Tuning - Quick Demo")
    print("=" * 40)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Run grid search demo
    tuner, results = demonstrate_grid_search()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_hyperparameter_demo() 