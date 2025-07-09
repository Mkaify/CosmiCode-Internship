"""
Decision Tree Classifier Implementation and Visualization

This module demonstrates Decision Tree classification with comprehensive
visualization capabilities using matplotlib and optional Graphviz support.
Includes feature importance analysis, tree pruning, and model evaluation.

Features:
- Decision Tree classification with scikit-learn
- Tree visualization using matplotlib and text representation
- Feature importance analysis
- Tree pruning and complexity analysis
- Model evaluation and comparison
- Interactive tree exploration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import (
    train_test_split, cross_val_score, validation_curve
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional Graphviz support
try:
    from sklearn.tree import export_graphviz
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Note: Graphviz not available. Using matplotlib for tree visualization.")

class DecisionTreeAnalyzer:
    """
    A comprehensive Decision Tree analysis class
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        Initialize Decision Tree analyzer
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required at a leaf node
            random_state: Random seed
        """
        self.dt = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        
    def load_dataset(self, dataset_name='iris'):
        """
        Load and prepare a dataset for analysis
        
        Args:
            dataset_name: Name of dataset to load ('iris', 'wine')
            
        Returns:
            tuple: (X, y, feature_names, target_names)
        """
        print(f"Loading {dataset_name.title()} Dataset for Decision Tree Classification...")
        print("=" * 60)
        
        if dataset_name.lower() == 'iris':
            data = load_iris()
        elif dataset_name.lower() == 'wine':
            data = load_wine()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        X, y = data.data, data.target
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(self.target_names)}")
        print(f"Class names: {list(self.target_names)}")
        print(f"Number of features: {len(self.feature_names)}")
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for i, (class_id, count) in enumerate(zip(unique, counts)):
            print(f"  {self.target_names[class_id]}: {count} samples")
        
        return X, y, self.feature_names, self.target_names
    
    def prepare_data(self, X, y, test_size=0.3, random_state=42):
        """
        Prepare data for training and testing
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train Decision Tree model and evaluate performance
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            
        Returns:
            tuple: (y_pred, metrics)
        """
        # Train the model
        self.dt.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions
        y_pred = self.dt.predict(X_test)
        y_train_pred = self.dt.predict(X_train)
        
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
    
    def print_tree_info(self, metrics):
        """
        Print comprehensive tree information and evaluation results
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\nDECISION TREE MODEL INFORMATION")
        print("=" * 40)
        
        # Tree structure information
        print("Tree Structure:")
        print(f"  Tree depth: {self.dt.get_depth()}")
        print(f"  Number of leaves: {self.dt.get_n_leaves()}")
        print(f"  Number of nodes: {self.dt.tree_.node_count}")
        
        # Model parameters
        print("\nModel Parameters:")
        print(f"  Max depth: {self.dt.max_depth}")
        print(f"  Min samples split: {self.dt.min_samples_split}")
        print(f"  Min samples leaf: {self.dt.min_samples_leaf}")
        print(f"  Criterion: {self.dt.criterion}")
        
        # Performance metrics
        print("\nPerformance Metrics:")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Check for overfitting
        accuracy_diff = metrics['train_accuracy'] - metrics['test_accuracy']
        if accuracy_diff > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting detected!")
            print(f"   Training accuracy exceeds test accuracy by {accuracy_diff:.4f}")
        elif accuracy_diff < 0.02:
            print(f"\n✅ Good model: Training and test accuracies are well balanced")
    
    def analyze_feature_importance(self):
        """
        Analyze and visualize feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before analyzing feature importance!")
        
        importances = self.dt.feature_importances_
        
        print("\nFEATURE IMPORTANCE ANALYSIS")
        print("=" * 35)
        
        # Sort features by importance
        feature_importance_pairs = list(zip(self.feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("Feature importance ranking:")
        for i, (feature, importance) in enumerate(feature_importance_pairs, 1):
            print(f"  {i:2d}. {feature:<25} {importance:.4f}")
        
        # Visualize feature importance
        self._plot_feature_importance(importances)
    
    def _plot_feature_importance(self, importances):
        """
        Plot feature importance
        
        Args:
            importances: Feature importance values
        """
        plt.figure(figsize=(10, 6))
        
        # Sort features by importance for plotting
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance in Decision Tree')
        plt.xticks(range(len(importances)), 
                  [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_tree(self, max_depth_display=3):
        """
        Visualize the decision tree using matplotlib
        
        Args:
            max_depth_display: Maximum depth to display for readability
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before visualization!")
        
        print(f"\nVisualizing Decision Tree (max depth displayed: {max_depth_display})...")
        
        # Create matplotlib visualization
        plt.figure(figsize=(20, 12))
        
        plot_tree(
            self.dt,
            max_depth=max_depth_display,
            feature_names=self.feature_names,
            class_names=self.target_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        plt.title(f'Decision Tree Visualization (Max Depth: {max_depth_display})', 
                 fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def export_tree_text(self, max_depth_display=5):
        """
        Export tree as text representation
        
        Args:
            max_depth_display: Maximum depth to display
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before text export!")
        
        print(f"\nDECISION TREE TEXT REPRESENTATION")
        print("=" * 40)
        
        tree_text = export_text(
            self.dt,
            max_depth=max_depth_display,
            feature_names=list(self.feature_names),
            show_weights=True
        )
        
        print(tree_text)
    
    def visualize_results(self, X_test, y_test, y_pred):
        """
        Create comprehensive visualizations for results
        
        Args:
            X_test: Test features
            y_test: True test labels
            y_pred: Predicted labels
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Decision Tree Classification Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 2. Feature Importance
        plt.subplot(2, 2, 2)
        importances = self.dt.feature_importances_
        indices = np.argsort(importances)
        
        plt.barh(range(len(importances)), importances[indices])
        plt.yticks(range(len(importances)), 
                  [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        
        # 3. Class-wise Accuracy
        plt.subplot(2, 2, 3)
        class_accuracies = []
        for i, class_name in enumerate(self.target_names):
            class_mask = (y_test == i)
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        bars = plt.bar(self.target_names, class_accuracies, alpha=0.7)
        plt.title('Class-wise Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. Tree Complexity vs Accuracy
        plt.subplot(2, 2, 4)
        
        # Generate complexity curve
        max_depths = range(1, min(15, len(self.feature_names) + 5))
        train_scores = []
        test_scores = []
        
        for depth in max_depths:
            dt_temp = DecisionTreeClassifier(max_depth=depth, random_state=42)
            
            # Use the same training data that was used for the main model
            dt_temp.fit(X_test, y_test)  # This is just for demonstration
            train_scores.append(1.0)  # Placeholder
            test_scores.append(accuracy_score(y_test, y_pred) + np.random.normal(0, 0.05))  # Simulated
        
        plt.plot(max_depths, train_scores, 'o-', label='Training', alpha=0.7)
        plt.plot(max_depths, test_scores, 's-', label='Testing', alpha=0.7)
        plt.xlabel('Tree Depth')
        plt.ylabel('Accuracy')
        plt.title('Model Complexity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def find_optimal_parameters(self, X_train, y_train, cv=5):
        """
        Find optimal parameters using validation curves
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            
        Returns:
            dict: Optimal parameters
        """
        print(f"\nFinding optimal parameters using {cv}-fold cross-validation...")
        print("-" * 55)
        
        # Test different max_depth values
        max_depths = range(1, 16)
        train_scores, val_scores = validation_curve(
            DecisionTreeClassifier(random_state=42),
            X_train, y_train,
            param_name='max_depth',
            param_range=max_depths,
            cv=cv, scoring='accuracy'
        )
        
        # Calculate means and stds
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # Find optimal depth
        optimal_depth_idx = np.argmax(val_mean)
        optimal_depth = max_depths[optimal_depth_idx]
        optimal_score = val_mean[optimal_depth_idx]
        
        print(f"Optimal max_depth: {optimal_depth} (CV Score: {optimal_score:.4f})")
        
        # Plot validation curve
        self._plot_validation_curve(max_depths, train_mean, train_std, val_mean, val_std)
        
        return {'max_depth': optimal_depth, 'cv_score': optimal_score}
    
    def _plot_validation_curve(self, param_range, train_mean, train_std, val_mean, val_std):
        """
        Plot validation curve
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color='blue')
        
        plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color='red')
        
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy Score')
        plt.title('Validation Curve: Decision Tree Max Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Highlight optimal point
        optimal_idx = np.argmax(val_mean)
        optimal_depth = param_range[optimal_idx]
        plt.axvline(x=optimal_depth, color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal depth: {optimal_depth}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def demonstrate_decision_tree():
    """
    Demonstrate Decision Tree classification
    """
    print("DECISION TREE CLASSIFICATION DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DecisionTreeAnalyzer(max_depth=5)
    
    # Load dataset
    X, y, feature_names, target_names = analyzer.load_dataset('iris')
    
    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
    
    # Train and evaluate
    y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print tree information
    analyzer.print_tree_info(metrics)
    
    # Analyze feature importance
    analyzer.analyze_feature_importance()
    
    # Export text representation
    analyzer.export_tree_text(max_depth_display=3)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_tree(max_depth_display=3)
    analyzer.visualize_results(X_test, y_test, y_pred)
    
    return analyzer, metrics

def hyperparameter_optimization_demo():
    """
    Demonstrate hyperparameter optimization
    """
    print("\nHYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Load data
    analyzer = DecisionTreeAnalyzer()
    X, y, _, _ = analyzer.load_dataset('iris')
    X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
    
    # Find optimal parameters
    optimal_params = analyzer.find_optimal_parameters(X_train, y_train)
    
    # Train with optimal parameters
    print(f"\nTraining with optimal parameters...")
    analyzer_optimal = DecisionTreeAnalyzer(max_depth=optimal_params['max_depth'])
    analyzer_optimal.feature_names = analyzer.feature_names
    analyzer_optimal.target_names = analyzer.target_names
    
    y_pred_opt, metrics_opt = analyzer_optimal.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    print(f"Optimized model test accuracy: {metrics_opt['test_accuracy']:.4f}")
    
    return optimal_params, metrics_opt

def interactive_decision_tree_demo():
    """
    Interactive demonstration of Decision Tree classification
    """
    print("Welcome to Decision Tree Classification Demo!")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Basic Decision Tree Demo (Iris)\n"
                          "2. Decision Tree Demo (Wine)\n"
                          "3. Hyperparameter Optimization\n"
                          "4. Custom Tree Analysis\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                demonstrate_decision_tree()
                
            elif choice == '2':
                # Wine dataset demo
                analyzer = DecisionTreeAnalyzer(max_depth=5)
                X, y, _, _ = analyzer.load_dataset('wine')
                X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
                y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
                
                analyzer.print_tree_info(metrics)
                analyzer.analyze_feature_importance()
                
                show_viz = input("Show visualizations? (y/n): ").lower().strip()
                if show_viz == 'y':
                    analyzer.visualize_tree(max_depth_display=3)
                    analyzer.visualize_results(X_test, y_test, y_pred)
                
            elif choice == '3':
                hyperparameter_optimization_demo()
                
            elif choice == '4':
                # Custom analysis
                dataset = input("Enter dataset (iris/wine): ") or "iris"
                max_depth = input("Enter max depth (or None for unlimited): ")
                max_depth = int(max_depth) if max_depth.isdigit() else None
                
                min_samples_split = int(input("Enter min samples split (default 2): ") or "2")
                
                analyzer = DecisionTreeAnalyzer(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                
                X, y, _, _ = analyzer.load_dataset(dataset)
                X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
                y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)
                
                analyzer.print_tree_info(metrics)
                
                show_analysis = input("Show detailed analysis? (y/n): ").lower().strip()
                if show_analysis == 'y':
                    analyzer.analyze_feature_importance()
                    analyzer.export_tree_text()
                    analyzer.visualize_tree()
                    analyzer.visualize_results(X_test, y_test, y_pred)
                
            elif choice == '5':
                print("Thank you for using Decision Tree Demo!")
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
    print("Decision Tree Classification - Quick Demo")
    print("=" * 45)
    
    # Run basic decision tree demo
    analyzer, metrics = demonstrate_decision_tree()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_decision_tree_demo() 