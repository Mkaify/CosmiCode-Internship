"""
K-Nearest Neighbors (KNN) Classifier Implementation

This module demonstrates K-Nearest Neighbors classification using the Iris dataset
with comprehensive evaluation, visualization, and parameter tuning capabilities.

Features:
- KNN classification with scikit-learn
- Accuracy evaluation and metrics
- Cross-validation analysis
- Hyperparameter tuning (k value)
- Decision boundary visualization
- Confusion matrix and classification report
- Feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    validation_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class KNNAnalyzer:
    """
    A comprehensive KNN analysis class
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski'):
        """
        Initialize KNN analyzer
        
        Args:
            n_neighbors: Number of neighbors to use
            weights: Weight function ('uniform', 'distance')
            metric: Distance metric to use
        """
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        
    def load_iris_dataset(self):
        """
        Load and prepare the Iris dataset
        
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        print("Loading Iris Dataset for KNN Classification...")
        print("=" * 45)
        
        iris = load_iris()
        X, y = iris.data, iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(self.target_names)}")
        print(f"Class names: {list(self.target_names)}")
        print(f"Features: {list(self.feature_names)}")
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for i, (class_id, count) in enumerate(zip(unique, counts)):
            print(f"  {self.target_names[class_id]}: {count} samples")
        
        return X, y, self.feature_names, self.target_names
    
    def prepare_data(self, X, y, test_size=0.3, random_state=42, scale_features=True):
        """
        Prepare data for training and testing
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            random_state: Random seed
            scale_features: Whether to scale features
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features scaled: {scale_features}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train KNN model and evaluate performance
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            tuple: (y_pred, metrics)
        """
        # Train the model
        self.knn.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.knn.predict(X_test)
        y_train_pred = self.knn.predict(X_train)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        return y_pred, metrics
    
    def print_evaluation_results(self, y_test, y_pred, metrics):
        """
        Print comprehensive evaluation results
        
        Args:
            y_test: True test labels
            y_pred: Predicted labels
            metrics: Dictionary of evaluation metrics
        """
        print("\nKNN MODEL EVALUATION RESULTS")
        print("=" * 35)
        
        # Basic metrics
        print("Performance Metrics:")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Model parameters
        print("\nModel Parameters:")
        print(f"  Number of neighbors (k): {self.knn.n_neighbors}")
        print(f"  Weights: {self.knn.weights}")
        print(f"  Distance metric: {self.knn.metric}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))
        
        # Confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    
    def visualize_results(self, X_train, X_test, y_train, y_test, y_pred):
        """
        Create comprehensive visualizations for KNN results
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            y_pred: Predictions
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('K-Nearest Neighbors Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 2. Feature pair visualization (using PCA if more than 2 features)
        plt.subplot(2, 2, 2)
        if X_train.shape[1] > 2:
            # Use PCA to reduce to 2D for visualization
            pca = PCA(n_components=2)
            X_train_2d = pca.fit_transform(X_train)
            X_test_2d = pca.transform(X_test)
            feature_labels = ['PC1', 'PC2']
        else:
            X_train_2d = X_train
            X_test_2d = X_test
            feature_labels = self.feature_names[:2]
        
        # Plot training data
        scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                            c=y_train, cmap='viridis', alpha=0.6, 
                            label='Training data')
        
        # Plot test data with different marker
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], 
                   c=y_test, cmap='viridis', marker='s', 
                   s=100, alpha=0.8, edgecolors='black',
                   label='Test data')
        
        plt.xlabel(feature_labels[0])
        plt.ylabel(feature_labels[1])
        plt.title('Data Distribution (2D Projection)')
        plt.legend()
        plt.colorbar(scatter)
        
        # 3. Class-wise accuracy
        plt.subplot(2, 2, 3)
        class_accuracies = []
        for i, class_name in enumerate(self.target_names):
            class_mask = (y_test == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        bars = plt.bar(self.target_names, class_accuracies)
        plt.title('Class-wise Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # 4. Prediction confidence (distance to neighbors)
        plt.subplot(2, 2, 4)
        distances, indices = self.knn.kneighbors(X_test)
        avg_distances = np.mean(distances, axis=1)
        
        plt.scatter(range(len(avg_distances)), avg_distances, 
                   c=y_pred, cmap='viridis', alpha=0.7)
        plt.xlabel('Test Sample Index')
        plt.ylabel('Average Distance to Neighbors')
        plt.title('Prediction Confidence')
        plt.colorbar(label='Predicted Class')
        
        plt.tight_layout()
        plt.show()
    
    def find_optimal_k(self, X_train, y_train, k_range=range(1, 31), cv=5):
        """
        Find optimal k value using cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            k_range: Range of k values to test
            cv: Number of cross-validation folds
            
        Returns:
            tuple: (optimal_k, scores)
        """
        print(f"\nFinding optimal k value using {cv}-fold cross-validation...")
        print(f"Testing k values from {min(k_range)} to {max(k_range)}")
        print("-" * 50)
        
        scores = []
        for k in k_range:
            knn_temp = KNeighborsClassifier(n_neighbors=k)
            cv_scores = cross_val_score(knn_temp, X_train, y_train, cv=cv, scoring='accuracy')
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            scores.append((k, mean_score, std_score))
            print(f"k={k:2d}: Accuracy = {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Find optimal k
        optimal_k = max(scores, key=lambda x: x[1])[0]
        optimal_score = max(scores, key=lambda x: x[1])[1]
        
        print(f"\nOptimal k: {optimal_k} (Accuracy: {optimal_score:.4f})")
        
        # Visualize k selection
        self._plot_k_selection(scores)
        
        return optimal_k, scores
    
    def _plot_k_selection(self, scores):
        """
        Plot k value selection results
        
        Args:
            scores: List of (k, mean_score, std_score) tuples
        """
        k_values = [score[0] for score in scores]
        mean_scores = [score[1] for score in scores]
        std_scores = [score[2] for score in scores]
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(k_values, mean_scores, yerr=std_scores, 
                    marker='o', capsize=5, capthick=2)
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Cross-validation Accuracy')
        plt.title('KNN Hyperparameter Tuning: Optimal k Selection')
        plt.grid(True, alpha=0.3)
        
        # Highlight optimal k
        optimal_k = max(scores, key=lambda x: x[1])[0]
        optimal_score = max(scores, key=lambda x: x[1])[1]
        plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal k={optimal_k}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

def demonstrate_knn_classification():
    """
    Demonstrate KNN classification on Iris dataset
    """
    print("KNN CLASSIFICATION DEMONSTRATION")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = KNNAnalyzer(n_neighbors=5)
    
    # Load dataset
    X, y, feature_names, target_names = analyzer.load_iris_dataset()
    
    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y, scale_features=True)
    
    # Train and evaluate
    y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print results
    analyzer.print_evaluation_results(y_test, y_pred, metrics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_results(X_train, X_test, y_train, y_test, y_pred)
    
    return analyzer, metrics

def hyperparameter_tuning_demo():
    """
    Demonstrate hyperparameter tuning for KNN
    """
    print("\nHYPERPARAMETER TUNING DEMONSTRATION")
    print("=" * 40)
    
    # Load data
    analyzer = KNNAnalyzer()
    X, y, _, _ = analyzer.load_iris_dataset()
    X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
    
    # Find optimal k
    optimal_k, scores = analyzer.find_optimal_k(X_train, y_train)
    
    # Test with optimal k
    print(f"\nTesting with optimal k={optimal_k}...")
    analyzer_optimal = KNNAnalyzer(n_neighbors=optimal_k)
    analyzer_optimal.feature_names = analyzer.feature_names
    analyzer_optimal.target_names = analyzer.target_names
    
    y_pred_opt, metrics_opt = analyzer_optimal.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print(f"Optimized model accuracy: {metrics_opt['test_accuracy']:.4f}")
    
    return optimal_k, metrics_opt

def compare_knn_configurations():
    """
    Compare different KNN configurations
    """
    print("\nKNN CONFIGURATION COMPARISON")
    print("=" * 35)
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    configurations = [
        ("k=3, uniform, unscaled", 3, 'uniform', X_train, X_test),
        ("k=3, uniform, scaled", 3, 'uniform', X_train_scaled, X_test_scaled),
        ("k=5, uniform, scaled", 5, 'uniform', X_train_scaled, X_test_scaled),
        ("k=5, distance, scaled", 5, 'distance', X_train_scaled, X_test_scaled),
        ("k=7, distance, scaled", 7, 'distance', X_train_scaled, X_test_scaled),
    ]
    
    results = []
    
    for name, k, weights, X_tr, X_te in configurations:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(X_tr, y_train)
        
        train_acc = knn.score(X_tr, y_train)
        test_acc = knn.score(X_te, y_test)
        
        results.append({
            'Configuration': name,
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Difference': train_acc - test_acc
        })
        
        print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")
    
    # Display comparison table
    print("\nCOMPARISON RESULTS:")
    print("=" * 60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

def interactive_knn_demo():
    """
    Interactive demonstration of KNN classification
    """
    print("Welcome to KNN Classification Demo!")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Basic KNN Classification Demo\n"
                          "2. Hyperparameter Tuning Demo\n"
                          "3. Configuration Comparison\n"
                          "4. Custom KNN Analysis\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                demonstrate_knn_classification()
                
            elif choice == '2':
                hyperparameter_tuning_demo()
                
            elif choice == '3':
                compare_knn_configurations()
                
            elif choice == '4':
                # Custom analysis
                k = int(input("Enter number of neighbors (1-20): ") or "5")
                weights = input("Enter weights (uniform/distance): ") or "uniform"
                scale = input("Scale features? (y/n): ").lower().strip() == 'y'
                
                analyzer = KNNAnalyzer(n_neighbors=k, weights=weights)
                X, y, _, _ = analyzer.load_iris_dataset()
                X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y, scale_features=scale)
                
                y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
                analyzer.print_evaluation_results(y_test, y_pred, metrics)
                
                visualize = input("Create visualizations? (y/n): ").lower().strip()
                if visualize == 'y':
                    analyzer.visualize_results(X_train, X_test, y_train, y_test, y_pred)
                
            elif choice == '5':
                print("Thank you for using KNN Classification Demo!")
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
    print("K-Nearest Neighbors Classification - Quick Demo")
    print("=" * 50)
    
    # Run basic KNN demo
    analyzer, metrics = demonstrate_knn_classification()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_knn_demo() 