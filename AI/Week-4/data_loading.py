"""
Data Loading and Exploration with Scikit-Learn

This module demonstrates how to load datasets using scikit-learn, specifically
focusing on the famous Iris dataset. It includes basic data exploration,
visualization, and statistical analysis.

Features:
- Load built-in datasets from scikit-learn
- Basic data exploration and statistics
- Data visualization
- Dataset information display
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

def load_iris_dataset():
    """
    Load and explore the Iris dataset
    
    Returns:
        tuple: (features, target, feature_names, target_names)
    """
    print("Loading Iris Dataset...")
    print("=" * 40)
    
    # Load the Iris dataset
    iris = load_iris()
    
    # Extract features and target
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Basic information
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Number of classes: {len(target_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Statistics:")
    print(df.describe())
    
    print("\nClass distribution:")
    print(df['species'].value_counts())
    
    return X, y, feature_names, target_names, df

def visualize_iris_dataset(df):
    """
    Create visualizations for the Iris dataset
    
    Args:
        df: DataFrame containing the Iris data
    """
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Exploration', fontsize=16, fontweight='bold')
    
    # 1. Pairplot of features
    plt.subplot(2, 2, 1)
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
    
    # Scatter plot of sepal length vs width
    for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], 
                   species_data['sepal width (cm)'], 
                   label=species, alpha=0.7)
    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Sepal Length vs Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Petal measurements
    plt.subplot(2, 2, 2)
    for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
        species_data = df[df['species'] == species]
        plt.scatter(species_data['petal length (cm)'], 
                   species_data['petal width (cm)'], 
                   label=species, alpha=0.7)
    
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.title('Petal Length vs Width')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Feature distributions
    plt.subplot(2, 2, 3)
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
    
    # Box plot for all features
    df_melted = df[feature_cols + ['species']].melt(id_vars=['species'], 
                                                   var_name='feature', 
                                                   value_name='value')
    
    # Create box plot
    box_data = []
    labels = []
    for feature in feature_cols:
        box_data.append(df[feature])
        labels.append(feature.split('(')[0].strip())
    
    plt.boxplot(box_data, labels=labels)
    plt.title('Feature Distributions')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    plt.subplot(2, 2, 4)
    correlation_matrix = df[feature_cols].corr()
    
    # Create heatmap
    im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
    
    # Set labels
    plt.xticks(range(len(feature_cols)), 
               [name.split('(')[0].strip() for name in feature_cols], 
               rotation=45)
    plt.yticks(range(len(feature_cols)), 
               [name.split('(')[0].strip() for name in feature_cols])
    
    # Add correlation values
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center')
    
    plt.title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.show()

def load_other_datasets():
    """
    Load and explore other built-in datasets from scikit-learn
    """
    print("\n" + "=" * 50)
    print("EXPLORING OTHER SCIKIT-LEARN DATASETS")
    print("=" * 50)
    
    datasets_info = [
        ("Wine Dataset", load_wine),
        ("Breast Cancer Dataset", load_breast_cancer)
    ]
    
    for name, load_func in datasets_info:
        print(f"\n{name}:")
        print("-" * 30)
        
        # Load dataset
        dataset = load_func()
        X, y = dataset.data, dataset.target
        
        print(f"Shape: {X.shape}")
        print(f"Features: {len(dataset.feature_names)}")
        print(f"Classes: {len(dataset.target_names)}")
        print(f"Target names: {dataset.target_names}")
        print(f"Description: {dataset.DESCR[:200]}...")

def compare_datasets():
    """
    Compare different datasets available in scikit-learn
    """
    print("\n" + "=" * 50)
    print("DATASET COMPARISON")
    print("=" * 50)
    
    # Load datasets
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    
    datasets = [
        ("Iris", iris),
        ("Wine", wine),
        ("Breast Cancer", cancer)
    ]
    
    # Create comparison table
    print(f"{'Dataset':<15} {'Samples':<10} {'Features':<10} {'Classes':<10} {'Type':<15}")
    print("-" * 70)
    
    for name, dataset in datasets:
        n_samples, n_features = dataset.data.shape
        n_classes = len(dataset.target_names)
        
        # Determine problem type
        if n_classes == 2:
            problem_type = "Binary Class."
        elif n_classes > 2:
            problem_type = "Multi Class."
        else:
            problem_type = "Regression"
        
        print(f"{name:<15} {n_samples:<10} {n_features:<10} {n_classes:<10} {problem_type:<15}")

def dataset_statistics(X, y, feature_names, target_names):
    """
    Calculate and display detailed statistics for a dataset
    
    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        target_names: List of target class names
    """
    print("\n" + "=" * 50)
    print("DETAILED DATASET STATISTICS")
    print("=" * 50)
    
    # Basic statistics
    print(f"Dataset dimensions: {X.shape}")
    print(f"Total samples: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")
    print(f"Total classes: {len(target_names)}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for i, (class_id, count) in enumerate(zip(unique, counts)):
        percentage = (count / len(y)) * 100
        print(f"  {target_names[class_id]}: {count} samples ({percentage:.1f}%)")
    
    # Feature statistics
    print(f"\nFeature statistics:")
    print(f"{'Feature':<25} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8}")
    print("-" * 65)
    
    for i, feature in enumerate(feature_names):
        min_val = np.min(X[:, i])
        max_val = np.max(X[:, i])
        mean_val = np.mean(X[:, i])
        std_val = np.std(X[:, i])
        
        print(f"{feature:<25} {min_val:<8.2f} {max_val:<8.2f} {mean_val:<8.2f} {std_val:<8.2f}")

def interactive_dataset_explorer():
    """
    Interactive dataset exploration tool
    """
    print("Welcome to Dataset Explorer!")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Load and explore Iris dataset\n"
                          "2. Load other datasets\n"
                          "3. Compare datasets\n"
                          "4. Detailed statistics\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                X, y, feature_names, target_names, df = load_iris_dataset()
                
                visualize = input("\nCreate visualizations? (y/n): ").lower().strip()
                if visualize == 'y':
                    visualize_iris_dataset(df)
                
                stats = input("Show detailed statistics? (y/n): ").lower().strip()
                if stats == 'y':
                    dataset_statistics(X, y, feature_names, target_names)
                    
            elif choice == '2':
                load_other_datasets()
                
            elif choice == '3':
                compare_datasets()
                
            elif choice == '4':
                # Load Iris for demonstration
                iris = load_iris()
                dataset_statistics(iris.data, iris.target, 
                                 iris.feature_names, iris.target_names)
                
            elif choice == '5':
                print("Thank you for using Dataset Explorer!")
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
    print("Scikit-Learn Dataset Loading - Quick Demo")
    print("=" * 45)
    
    # Load and display basic info about Iris dataset
    X, y, feature_names, target_names, df = load_iris_dataset()
    
    # Show basic visualization
    print("\nCreating basic visualization...")
    visualize_iris_dataset(df)
    
    print("\n" + "=" * 50)
    
    # Run interactive explorer
    interactive_dataset_explorer() 