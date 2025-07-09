# Week 4: Machine Learning Basics for AI (Intermediate to Advanced)

This folder contains comprehensive implementations of fundamental machine learning algorithms and techniques using scikit-learn. The programs demonstrate data loading, preprocessing, model training, evaluation, and visualization.

## 📁 Files Overview

| File | Description |
|------|-------------|
| `data_loading.py` | Dataset loading and exploration with scikit-learn |
| `data_preprocessing.py` | Data preprocessing techniques and missing value handling |
| `linear_regression.py` | Linear regression implementation with visualization |
| `knn_classifier.py` | K-Nearest Neighbors classifier on Iris dataset |
| `decision_tree.py` | Decision Tree classifier with visualization |
| `requirements.txt` | Python package dependencies |
| `README.md` | This documentation file |

## 🎯 Learning Objectives

By the end of this week, you will understand:

1. **Data Loading**: Using scikit-learn's built-in datasets
2. **Data Preprocessing**: Handling missing values and categorical encoding
3. **Regression Analysis**: Linear regression with evaluation metrics
4. **Classification**: KNN and Decision Tree algorithms
5. **Model Evaluation**: Accuracy, precision, recall, and visualization
6. **Hyperparameter Tuning**: Finding optimal model parameters

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Basic understanding of machine learning concepts

### Installation

1. Navigate to the Week-4 directory
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For enhanced tree visualization, install Graphviz:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install graphviz
   
   # On macOS
   brew install graphviz
   
   # On Windows - download from https://graphviz.org/download/
   
   # Then install Python package
   pip install graphviz
   ```

## 📊 Implementation Details

### 1. Data Loading and Exploration (`data_loading.py`)

**Purpose:** Learn to load and explore datasets using scikit-learn

**Key Features:**
- Load built-in datasets (Iris, Wine, Breast Cancer)
- Basic data exploration and statistics
- Data visualization with matplotlib and seaborn
- Dataset comparison and analysis

**Run the program:**
```bash
python data_loading.py
```

**What you'll learn:**
- Dataset structure and properties
- Feature and target analysis
- Data visualization techniques
- Statistical summaries

### 2. Data Preprocessing (`data_preprocessing.py`)

**Purpose:** Master essential data preprocessing techniques

**Key Features:**
- Missing value detection and handling
- Categorical data encoding (Label, One-Hot, Target)
- Feature scaling and normalization
- Outlier detection and analysis
- Complete preprocessing pipeline

**Run the program:**
```bash
python data_preprocessing.py
```

**Preprocessing Techniques:**
- **Missing Values**: Mean/median/mode imputation, KNN imputation
- **Categorical Encoding**: Label encoding, one-hot encoding
- **Feature Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Outlier Detection**: IQR method, Z-score method

### 3. Linear Regression (`linear_regression.py`)

**Purpose:** Implement and analyze linear regression models

**Key Features:**
- Simple and multiple linear regression
- Model evaluation with comprehensive metrics
- Residual analysis and diagnostic plots
- Feature importance analysis
- Visualization of regression results

**Run the program:**
```bash
python linear_regression.py
```

**Analysis Includes:**
- **Metrics**: MSE, RMSE, MAE, R², Explained Variance
- **Visualizations**: Scatter plots, residual plots, prediction vs actual
- **Model Interpretation**: Coefficients, equation, performance assessment

### 4. K-Nearest Neighbors (`knn_classifier.py`)

**Purpose:** Implement KNN classification with the Iris dataset

**Key Features:**
- KNN classification with accuracy evaluation
- Hyperparameter tuning (optimal k selection)
- Cross-validation analysis
- Confusion matrix and classification metrics
- Decision boundary visualization

**Run the program:**
```bash
python knn_classifier.py
```

**Evaluation Metrics:**
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Detailed performance metrics
- **Confusion Matrix**: Class-wise performance analysis
- **Cross-validation**: Robust model assessment

### 5. Decision Tree Classifier (`decision_tree.py`)

**Purpose:** Build and visualize decision tree classifiers

**Key Features:**
- Decision tree classification
- Tree visualization using matplotlib
- Feature importance analysis
- Hyperparameter optimization
- Model complexity analysis

**Run the program:**
```bash
python decision_tree.py
```

**Visualization Options:**
- **Tree Structure**: Visual representation of decision nodes
- **Feature Importance**: Ranking of feature contributions
- **Text Export**: Human-readable tree rules
- **Performance Analysis**: Accuracy and complexity trade-offs

## 🎮 Interactive Examples

### Example 1: Quick Data Loading
```python
from data_loading import load_iris_dataset

# Load and explore Iris dataset
X, y, feature_names, target_names, df = load_iris_dataset()
print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
```

### Example 2: Data Preprocessing Pipeline
```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Create sample data with missing values
df = preprocessor.create_sample_dataset_with_missing_values()

# Handle missing values
df_clean = preprocessor.handle_missing_values(df, strategy='mean')

# Encode categorical data
df_encoded = preprocessor.encode_categorical_data(df_clean, encoding_type='label')
```

### Example 3: Linear Regression Analysis
```python
from linear_regression import LinearRegressionAnalyzer

# Initialize analyzer
analyzer = LinearRegressionAnalyzer()

# Create sample dataset
X, y, feature_names = analyzer.create_sample_dataset(n_features=1)

# Fit model and evaluate
X_train, X_test, y_train, y_test = analyzer.fit_model(X, y)
y_pred, metrics = analyzer.predict_and_evaluate(X_test, y_test)

print(f"R² Score: {metrics['r2']:.4f}")
```

### Example 4: KNN Classification
```python
from knn_classifier import KNNAnalyzer

# Initialize KNN analyzer
analyzer = KNNAnalyzer(n_neighbors=5)

# Load Iris dataset
X, y, feature_names, target_names = analyzer.load_iris_dataset()

# Train and evaluate
X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)

print(f"Accuracy: {metrics['test_accuracy']:.4f}")
```

### Example 5: Decision Tree Visualization
```python
from decision_tree import DecisionTreeAnalyzer

# Initialize Decision Tree
analyzer = DecisionTreeAnalyzer(max_depth=3)

# Load dataset and train
X, y, _, _ = analyzer.load_dataset('iris')
X_train, X_test, y_train, y_test = analyzer.prepare_data(X, y)
y_pred, metrics = analyzer.train_and_evaluate(X_train, X_test, y_train, y_test)

# Visualize tree and analyze importance
analyzer.visualize_tree(max_depth_display=3)
analyzer.analyze_feature_importance()
```

## 📚 Algorithm Comparison

### Performance Characteristics

| Algorithm | Pros | Cons | Best Use Cases |
|-----------|------|------|----------------|
| **Linear Regression** | Simple, interpretable, fast | Assumes linear relationships | Continuous target, feature analysis |
| **K-Nearest Neighbors** | Non-parametric, simple concept | Computationally expensive, sensitive to scale | Non-linear patterns, local structures |
| **Decision Trees** | Highly interpretable, handles non-linearity | Prone to overfitting, unstable | Rule extraction, feature selection |

### Complexity Analysis

| Algorithm | Training Time | Prediction Time | Space Complexity |
|-----------|---------------|-----------------|------------------|
| Linear Regression | O(n³) | O(1) | O(n) |
| KNN | O(1) | O(kn) | O(n) |
| Decision Tree | O(n log n) | O(log n) | O(n) |

## 🧪 Exercises and Challenges

### Basic Exercises
1. Compare different preprocessing strategies on the same dataset
2. Implement feature selection for linear regression
3. Tune KNN hyperparameters (k, weights, distance metrics)
4. Experiment with different tree pruning strategies

### Advanced Challenges
1. Create ensemble methods combining multiple algorithms
2. Implement custom distance metrics for KNN
3. Build regression trees for continuous targets
4. Develop automated preprocessing pipelines

### Real-world Projects
1. Predict house prices using regression
2. Build a customer classification system
3. Create a medical diagnosis decision tree
4. Develop a recommendation system using KNN

## ⚡ Performance Tips

### Data Preprocessing
1. **Handle missing values appropriately** for your domain
2. **Scale features** for distance-based algorithms (KNN)
3. **Encode categorical variables** properly
4. **Remove outliers** when they're genuine errors

### Model Training
1. **Use cross-validation** for robust evaluation
2. **Tune hyperparameters** systematically
3. **Monitor overfitting** with train/test accuracy gaps
4. **Validate assumptions** (e.g., linearity for regression)

### Visualization
1. **Always plot your data** before modeling
2. **Visualize predictions vs actuals** to check fit
3. **Use residual plots** to validate model assumptions
4. **Create confusion matrices** for classification problems

## 🐛 Common Pitfalls

1. **Data Leakage**: Using future information to predict past events
2. **Overfitting**: Model memorizes training data but fails on new data
3. **Underfitting**: Model is too simple to capture underlying patterns
4. **Feature Scaling**: Forgetting to scale features for distance-based algorithms
5. **Cross-validation**: Using incorrect CV strategies or data splitting

## 📈 Next Steps

After mastering these fundamentals, explore:
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Neural Networks**: Deep learning with TensorFlow/PyTorch
- **Unsupervised Learning**: Clustering, dimensionality reduction
- **Advanced Preprocessing**: Feature engineering, text processing
- **Model Deployment**: Making models production-ready

## 🤝 Contributing

Feel free to:
- Add more datasets and examples
- Implement additional algorithms
- Create more comprehensive visualizations
- Develop automated testing suites

## 📄 License

This educational content is provided for learning purposes. Feel free to use and modify for educational goals.

---

**Happy Machine Learning! 🤖**

*Remember: The key to successful machine learning is understanding your data, choosing appropriate algorithms, and validating your results thoroughly. These implementations provide a solid foundation for more advanced ML concepts.*

## 🔧 Installation Quick Start

```bash
# Clone/navigate to Week-4 directory
cd Week-4

# Install dependencies
pip install -r requirements.txt

# Run any program
python data_loading.py
python linear_regression.py
python knn_classifier.py
python decision_tree.py
```

## 📊 Sample Output

Each program provides:
- ✅ **Interactive menus** for guided exploration
- 📈 **Comprehensive visualizations** for better understanding  
- 📋 **Detailed metrics** and performance analysis
- 🔍 **Step-by-step explanations** of algorithms and results

Start with any program and follow the interactive prompts to explore machine learning concepts hands-on! 