"""
Linear Regression Implementation and Visualization

This module demonstrates linear regression using scikit-learn with comprehensive
visualization, evaluation metrics, and analysis. Includes both simple and
multiple linear regression examples.

Features:
- Simple and Multiple Linear Regression
- Model evaluation and metrics
- Comprehensive visualizations
- Residual analysis
- Feature importance analysis
- Custom dataset generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)
from sklearn.datasets import make_regression, load_diabetes
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LinearRegressionAnalyzer:
    """
    A comprehensive linear regression analysis class
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def create_sample_dataset(self, n_samples=100, n_features=1, noise=10, random_state=42):
        """
        Create a sample regression dataset
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            noise: Amount of noise to add
            random_state: Random seed
            
        Returns:
            tuple: (X, y, feature_names)
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        
        # Create feature names
        if n_features == 1:
            feature_names = ['feature']
        else:
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
        
        self.feature_names = feature_names
        
        return X, y, feature_names
    
    def load_real_dataset(self, dataset_name='diabetes'):
        """
        Load a real dataset for regression analysis
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            tuple: (X, y, feature_names)
        """
        if dataset_name == 'diabetes':
            data = load_diabetes()
            X, y = data.data, data.target
            feature_names = data.feature_names
        else:
            # Create synthetic data as fallback
            X, y, feature_names = self.create_sample_dataset(n_features=5)
        
        self.feature_names = feature_names
        return X, y, feature_names
    
    def fit_model(self, X, y, test_size=0.2, random_state=42, scale_features=False):
        """
        Fit the linear regression model
        
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
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        return X_train, X_test, y_train, y_test
    
    def predict_and_evaluate(self, X_test, y_test):
        """
        Make predictions and calculate evaluation metrics
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            tuple: (predictions, metrics)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'explained_variance': explained_variance_score(y_test, y_pred)
        }
        
        return y_pred, metrics
    
    def print_model_summary(self, metrics):
        """
        Print a comprehensive model summary
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        print("\nMODEL SUMMARY")
        print("=" * 30)
        
        # Model parameters
        print("Model Parameters:")
        if hasattr(self.model, 'coef_'):
            if len(self.model.coef_) == 1:
                print(f"  Coefficient: {self.model.coef_[0]:.4f}")
            else:
                print("  Coefficients:")
                for i, coef in enumerate(self.model.coef_):
                    feature_name = self.feature_names[i] if self.feature_names else f"Feature_{i+1}"
                    print(f"    {feature_name}: {coef:.4f}")
        
        print(f"  Intercept: {self.model.intercept_:.4f}")
        
        # Model equation
        print("\nModel Equation:")
        if len(self.model.coef_) == 1:
            print(f"  y = {self.model.coef_[0]:.4f} * x + {self.model.intercept_:.4f}")
        else:
            equation = f"y = {self.model.intercept_:.4f}"
            for i, coef in enumerate(self.model.coef_):
                feature_name = self.feature_names[i] if self.feature_names else f"x{i+1}"
                equation += f" + {coef:.4f} * {feature_name}"
            print(f"  {equation}")
        
        # Evaluation metrics
        print("\nEvaluation Metrics:")
        print(f"  Mean Squared Error (MSE): {metrics['mse']:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
        print(f"  Mean Absolute Error (MAE): {metrics['mae']:.4f}")
        print(f"  R-squared (R²): {metrics['r2']:.4f}")
        print(f"  Explained Variance Score: {metrics['explained_variance']:.4f}")
        
        # Interpretation
        print("\nModel Interpretation:")
        r2 = metrics['r2']
        if r2 >= 0.9:
            print("  Excellent fit - explains >90% of variance")
        elif r2 >= 0.7:
            print("  Good fit - explains >70% of variance")
        elif r2 >= 0.5:
            print("  Moderate fit - explains >50% of variance")
        else:
            print("  Poor fit - explains <50% of variance")
    
    def visualize_results(self, X_train, X_test, y_train, y_test, y_pred):
        """
        Create comprehensive visualizations for regression results
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
            y_pred: Predictions
        """
        # Determine if simple or multiple regression
        is_simple = X_train.shape[1] == 1
        
        if is_simple:
            self._visualize_simple_regression(X_train, X_test, y_train, y_test, y_pred)
        else:
            self._visualize_multiple_regression(X_train, X_test, y_train, y_test, y_pred)
    
    def _visualize_simple_regression(self, X_train, X_test, y_train, y_test, y_pred):
        """
        Visualizations for simple linear regression
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Simple Linear Regression Analysis', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot with regression line
        plt.subplot(2, 2, 1)
        plt.scatter(X_train, y_train, alpha=0.6, label='Training data', color='blue')
        plt.scatter(X_test, y_test, alpha=0.6, label='Test data', color='red')
        
        # Plot regression line
        X_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
        y_range_pred = self.model.predict(X_range)
        plt.plot(X_range, y_range_pred, 'g-', linewidth=2, label='Regression line')
        
        plt.xlabel('Feature Value')
        plt.ylabel('Target Value')
        plt.title('Regression Line Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Predictions vs Actual
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        plt.subplot(2, 2, 3)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # 4. Residuals histogram
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_multiple_regression(self, X_train, X_test, y_train, y_test, y_pred):
        """
        Visualizations for multiple linear regression
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multiple Linear Regression Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature importance (coefficients)
        plt.subplot(2, 2, 1)
        feature_names = self.feature_names if self.feature_names else [f'Feature_{i+1}' for i in range(len(self.model.coef_))]
        
        plt.barh(feature_names, self.model.coef_)
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance (Coefficients)')
        plt.grid(True, alpha=0.3)
        
        # 2. Predictions vs Actual
        plt.subplot(2, 2, 2)
        plt.scatter(y_test, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        plt.subplot(2, 2, 3)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # 4. Residuals histogram
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demonstrate_simple_regression():
    """
    Demonstrate simple linear regression
    """
    print("SIMPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 45)
    
    # Initialize analyzer
    analyzer = LinearRegressionAnalyzer()
    
    # Create simple dataset
    X, y, feature_names = analyzer.create_sample_dataset(n_samples=100, n_features=1, noise=15)
    
    print("Dataset created:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Fit model
    X_train, X_test, y_train, y_test = analyzer.fit_model(X, y)
    
    # Make predictions and evaluate
    y_pred, metrics = analyzer.predict_and_evaluate(X_test, y_test)
    
    # Print summary
    analyzer.print_model_summary(metrics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_results(X_train, X_test, y_train, y_test, y_pred)
    
    return analyzer, metrics

def demonstrate_multiple_regression():
    """
    Demonstrate multiple linear regression
    """
    print("\nMULTIPLE LINEAR REGRESSION DEMONSTRATION")
    print("=" * 47)
    
    # Initialize analyzer
    analyzer = LinearRegressionAnalyzer()
    
    # Load real dataset
    X, y, feature_names = analyzer.load_real_dataset('diabetes')
    
    print("Dataset loaded:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Feature names: {feature_names}")
    
    # Fit model with feature scaling
    X_train, X_test, y_train, y_test = analyzer.fit_model(X, y, scale_features=True)
    
    # Make predictions and evaluate
    y_pred, metrics = analyzer.predict_and_evaluate(X_test, y_test)
    
    # Print summary
    analyzer.print_model_summary(metrics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_results(X_train, X_test, y_train, y_test, y_pred)
    
    return analyzer, metrics

def compare_regression_models():
    """
    Compare different regression configurations
    """
    print("\nREGRESSION MODEL COMPARISON")
    print("=" * 35)
    
    configurations = [
        ("Simple (1 feature)", 1, False),
        ("Multiple (5 features)", 5, False),
        ("Multiple (5 features, scaled)", 5, True),
        ("Complex (10 features)", 10, True)
    ]
    
    results = []
    
    for name, n_features, scale in configurations:
        print(f"\nTesting: {name}")
        print("-" * 30)
        
        analyzer = LinearRegressionAnalyzer()
        X, y, _ = analyzer.create_sample_dataset(n_features=n_features, noise=20)
        
        X_train, X_test, y_train, y_test = analyzer.fit_model(X, y, scale_features=scale)
        y_pred, metrics = analyzer.predict_and_evaluate(X_test, y_test)
        
        results.append({
            'Configuration': name,
            'R²': metrics['r2'],
            'RMSE': metrics['rmse'],
            'Features': n_features,
            'Scaled': scale
        })
        
        print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    # Display comparison table
    print("\nCOMPARISON RESULTS:")
    print("=" * 60)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

def interactive_regression_demo():
    """
    Interactive demonstration of linear regression
    """
    print("Welcome to Linear Regression Demo!")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Simple Linear Regression Demo\n"
                          "2. Multiple Linear Regression Demo\n"
                          "3. Model Comparison\n"
                          "4. Custom Dataset Analysis\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                demonstrate_simple_regression()
                
            elif choice == '2':
                demonstrate_multiple_regression()
                
            elif choice == '3':
                compare_regression_models()
                
            elif choice == '4':
                # Custom analysis
                n_samples = int(input("Enter number of samples (50-500): ") or "100")
                n_features = int(input("Enter number of features (1-20): ") or "1")
                noise = float(input("Enter noise level (0-50): ") or "15")
                
                analyzer = LinearRegressionAnalyzer()
                X, y, _ = analyzer.create_sample_dataset(n_samples, n_features, noise)
                
                scale = input("Scale features? (y/n): ").lower().strip() == 'y'
                X_train, X_test, y_train, y_test = analyzer.fit_model(X, y, scale_features=scale)
                
                y_pred, metrics = analyzer.predict_and_evaluate(X_test, y_test)
                analyzer.print_model_summary(metrics)
                
                visualize = input("Create visualizations? (y/n): ").lower().strip()
                if visualize == 'y':
                    analyzer.visualize_results(X_train, X_test, y_train, y_test, y_pred)
                
            elif choice == '5':
                print("Thank you for using Linear Regression Demo!")
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
    print("Linear Regression Analysis - Quick Demo")
    print("=" * 42)
    
    # Run simple regression demo
    analyzer, metrics = demonstrate_simple_regression()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_regression_demo() 