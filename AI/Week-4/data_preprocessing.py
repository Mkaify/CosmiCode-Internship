"""
Data Preprocessing for Machine Learning

This module demonstrates various data preprocessing techniques essential for
machine learning, including handling missing values, encoding categorical data,
feature scaling, and data transformation.

Features:
- Missing value detection and handling
- Categorical data encoding (Label, One-Hot, Target)
- Feature scaling and normalization
- Data splitting and validation
- Outlier detection and handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler,
    RobustScaler
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, make_classification
import warnings
warnings.filterwarnings('ignore')

# Try to import TargetEncoder (available in sklearn >= 1.2.0)
try:
    from sklearn.preprocessing import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TARGET_ENCODER_AVAILABLE = False
    print("Note: TargetEncoder not available in this sklearn version. Using custom implementation.")

class DataPreprocessor:
    """
    A comprehensive data preprocessing class
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.imputer = None
        
    def create_sample_dataset_with_missing_values(self):
        """
        Create a sample dataset with missing values and categorical data
        
        Returns:
            pd.DataFrame: Sample dataset with missing values
        """
        # Load iris dataset as base
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = iris.target_names[iris.target]
        
        # Add some categorical features
        np.random.seed(42)
        df['color'] = np.random.choice(['red', 'blue', 'green'], size=len(df))
        df['size'] = np.random.choice(['small', 'medium', 'large'], size=len(df))
        df['season'] = np.random.choice(['spring', 'summer', 'fall', 'winter'], size=len(df))
        
        # Introduce missing values randomly
        missing_fraction = 0.15
        for column in df.columns:
            if column not in ['species']:  # Don't add missing values to target
                n_missing = int(len(df) * missing_fraction * np.random.random())
                missing_idx = np.random.choice(df.index, n_missing, replace=False)
                df.loc[missing_idx, column] = np.nan
        
        return df
    
    def analyze_missing_values(self, df):
        """
        Analyze missing values in the dataset
        
        Args:
            df: Input DataFrame
        
        Returns:
            pd.DataFrame: Missing value analysis
        """
        print("Missing Value Analysis")
        print("=" * 30)
        
        missing_data = []
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            data_type = str(df[column].dtype)
            
            missing_data.append({
                'Column': column,
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_percentage,
                'Data_Type': data_type
            })
        
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df.to_string(index=False))
        
        # Visualize missing values
        self.visualize_missing_values(df)
        
        return missing_df
    
    def visualize_missing_values(self, df):
        """
        Create visualizations for missing values
        
        Args:
            df: Input DataFrame
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing value heatmap
        plt.subplot(1, 2, 1)
        sns.heatmap(df.isnull(), cmap='viridis', cbar=True, yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.xlabel('Features')
        
        # Missing value bar plot
        plt.subplot(1, 2, 2)
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) > 0:
            missing_counts.plot(kind='bar')
            plt.title('Missing Values by Feature')
            plt.xlabel('Features')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Missing Values by Feature')
        
        plt.tight_layout()
        plt.show()
    
    def handle_missing_values(self, df, strategy='mean', categorical_strategy='most_frequent'):
        """
        Handle missing values using various strategies
        
        Args:
            df: Input DataFrame
            strategy: Strategy for numerical features ('mean', 'median', 'mode', 'drop', 'knn')
            categorical_strategy: Strategy for categorical features ('most_frequent', 'drop')
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        print(f"\nHandling Missing Values...")
        print(f"Numerical strategy: {strategy}")
        print(f"Categorical strategy: {categorical_strategy}")
        print("-" * 40)
        
        df_processed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Handle numerical missing values
        if numerical_cols:
            if strategy == 'drop':
                df_processed = df_processed.dropna(subset=numerical_cols)
            elif strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
                self.imputer = imputer
            else:
                imputer = SimpleImputer(strategy=strategy)
                df_processed[numerical_cols] = imputer.fit_transform(df_processed[numerical_cols])
                self.imputer = imputer
        
        # Handle categorical missing values
        if categorical_cols:
            if categorical_strategy == 'drop':
                df_processed = df_processed.dropna(subset=categorical_cols)
            else:
                imputer = SimpleImputer(strategy=categorical_strategy)
                df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
        
        # Report results
        original_missing = df.isnull().sum().sum()
        processed_missing = df_processed.isnull().sum().sum()
        
        print(f"Original missing values: {original_missing}")
        print(f"Remaining missing values: {processed_missing}")
        print(f"Rows before: {len(df)}, Rows after: {len(df_processed)}")
        
        return df_processed
    
    def encode_categorical_data(self, df, encoding_type='label'):
        """
        Encode categorical data using various techniques
        
        Args:
            df: Input DataFrame
            encoding_type: Type of encoding ('label', 'onehot', 'target')
        
        Returns:
            pd.DataFrame: DataFrame with encoded categorical data
        """
        print(f"\nEncoding Categorical Data using {encoding_type} encoding...")
        print("-" * 50)
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        if not categorical_cols:
            print("No categorical columns found!")
            return df_encoded
        
        print(f"Categorical columns found: {categorical_cols}")
        
        if encoding_type == 'label':
            # Label Encoding
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} unique values -> {list(range(len(le.classes_)))}")
        
        elif encoding_type == 'onehot':
            # One-Hot Encoding
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
            print(f"  Created {len(df_encoded.columns) - len(df.columns)} new binary features")
        
        elif encoding_type == 'target':
            # Target Encoding (requires target variable)
            if 'species' in categorical_cols:
                target_col = 'species'
                feature_cols = [col for col in categorical_cols if col != target_col]
                
                if TARGET_ENCODER_AVAILABLE:
                    # Use sklearn's TargetEncoder if available
                    # Create numeric target for encoding
                    if df_encoded[target_col].dtype == 'object':
                        le_target = LabelEncoder()
                        numeric_target = le_target.fit_transform(df_encoded[target_col])
                    else:
                        numeric_target = df_encoded[target_col]
                    
                    target_encoder = TargetEncoder()
                    df_encoded[feature_cols] = target_encoder.fit_transform(
                        df_encoded[feature_cols], numeric_target
                    )
                    print(f"  Encoded {len(feature_cols)} features using sklearn TargetEncoder")
                else:
                    # Custom target encoding implementation
                    # Create numeric target for encoding
                    if df_encoded[target_col].dtype == 'object':
                        le_target = LabelEncoder()
                        numeric_target = le_target.fit_transform(df_encoded[target_col])
                        df_encoded_temp = df_encoded.copy()
                        df_encoded_temp[target_col + '_numeric'] = numeric_target
                    else:
                        numeric_target = df_encoded[target_col]
                        df_encoded_temp = df_encoded.copy()
                        df_encoded_temp[target_col + '_numeric'] = numeric_target
                    
                    for col in feature_cols:
                        # Calculate mean target value for each category
                        target_means = df_encoded_temp.groupby(col)[target_col + '_numeric'].mean()
                        df_encoded[col] = df_encoded[col].map(target_means)
                        print(f"  {col}: Encoded using custom target means")
            else:
                print("  No target column 'species' found for target encoding")
        
        print(f"Shape before encoding: {df.shape}")
        print(f"Shape after encoding: {df_encoded.shape}")
        
        return df_encoded
    
    def scale_features(self, df, scaling_method='standard'):
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            scaling_method: Scaling method ('standard', 'minmax', 'robust')
        
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        print(f"\nScaling Features using {scaling_method} scaling...")
        print("-" * 40)
        
        df_scaled = df.copy()
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            print("No numerical columns to scale!")
            return df_scaled
        
        # Choose scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            print(f"Unknown scaling method: {scaling_method}")
            return df_scaled
        
        # Fit and transform
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        self.scaler = scaler
        
        print(f"Scaled {len(numerical_cols)} numerical features")
        print(f"Features scaled: {numerical_cols}")
        
        # Show scaling statistics
        print("\nScaling Statistics:")
        print(f"{'Feature':<25} {'Original Mean':<15} {'Scaled Mean':<15} {'Original Std':<15} {'Scaled Std':<15}")
        print("-" * 90)
        
        for col in numerical_cols:
            orig_mean = df[col].mean()
            orig_std = df[col].std()
            scaled_mean = df_scaled[col].mean()
            scaled_std = df_scaled[col].std()
            
            print(f"{col:<25} {orig_mean:<15.3f} {scaled_mean:<15.3f} {orig_std:<15.3f} {scaled_std:<15.3f}")
        
        return df_scaled
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """
        Detect outliers in numerical data
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            pd.DataFrame: DataFrame with outlier information
        """
        print(f"\nDetecting Outliers using {method} method...")
        print("-" * 40)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = []
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
            
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            outlier_info.append({
                'Feature': col,
                'Outlier_Count': outlier_count,
                'Outlier_Percentage': outlier_percentage,
                'Total_Values': len(df)
            })
        
        outlier_df = pd.DataFrame(outlier_info)
        print(outlier_df.to_string(index=False))
        
        return outlier_df

def demonstrate_preprocessing_pipeline():
    """
    Demonstrate a complete preprocessing pipeline
    """
    print("COMPLETE DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Step 1: Create sample dataset with issues
    print("Step 1: Creating sample dataset with missing values...")
    df_original = preprocessor.create_sample_dataset_with_missing_values()
    print(f"Original dataset shape: {df_original.shape}")
    print("\nFirst 5 rows:")
    print(df_original.head())
    
    # Step 2: Analyze missing values
    print("\nStep 2: Analyzing missing values...")
    missing_analysis = preprocessor.analyze_missing_values(df_original)
    
    # Step 3: Handle missing values
    print("\nStep 3: Handling missing values...")
    df_no_missing = preprocessor.handle_missing_values(df_original, strategy='mean')
    
    # Step 4: Encode categorical data
    print("\nStep 4: Encoding categorical data...")
    df_encoded = preprocessor.encode_categorical_data(df_no_missing, encoding_type='label')
    
    # Step 5: Scale features
    print("\nStep 5: Scaling features...")
    df_scaled = preprocessor.scale_features(df_encoded, scaling_method='standard')
    
    # Step 6: Detect outliers
    print("\nStep 6: Detecting outliers...")
    outlier_analysis = preprocessor.detect_outliers(df_scaled, method='iqr')
    
    # Final comparison
    print("\nFINAL COMPARISON:")
    print("=" * 30)
    print(f"Original shape: {df_original.shape}")
    print(f"Final shape: {df_scaled.shape}")
    print(f"Missing values removed: {df_original.isnull().sum().sum()}")
    print(f"Categorical features encoded: {len(df_original.select_dtypes(include=['object']).columns)}")
    
    return df_original, df_scaled

def interactive_preprocessing_demo():
    """
    Interactive demonstration of preprocessing techniques
    """
    print("Welcome to Data Preprocessing Demo!")
    print("=" * 40)
    
    preprocessor = DataPreprocessor()
    df = None
    
    while True:
        try:
            if df is None:
                choice = input("\nChoose an option:\n"
                              "1. Create sample dataset\n"
                              "2. Run complete pipeline\n"
                              "3. Exit\n"
                              "Enter your choice (1-3): ").strip()
            else:
                choice = input("\nChoose an option:\n"
                              "1. Analyze missing values\n"
                              "2. Handle missing values\n"
                              "3. Encode categorical data\n"
                              "4. Scale features\n"
                              "5. Detect outliers\n"
                              "6. Start over\n"
                              "7. Exit\n"
                              "Enter your choice (1-7): ").strip()
            
            if choice == '1' and df is None:
                df = preprocessor.create_sample_dataset_with_missing_values()
                print("Sample dataset created!")
                print(f"Shape: {df.shape}")
                print(df.head())
                
            elif choice == '1' and df is not None:
                preprocessor.analyze_missing_values(df)
                
            elif choice == '2' and df is None:
                demonstrate_preprocessing_pipeline()
                
            elif choice == '2' and df is not None:
                strategy = input("Enter strategy (mean/median/mode/knn): ") or 'mean'
                df = preprocessor.handle_missing_values(df, strategy=strategy)
                
            elif choice == '3' and df is not None:
                encoding = input("Enter encoding type (label/onehot): ") or 'label'
                df = preprocessor.encode_categorical_data(df, encoding_type=encoding)
                
            elif choice == '4' and df is not None:
                scaling = input("Enter scaling method (standard/minmax/robust): ") or 'standard'
                df = preprocessor.scale_features(df, scaling_method=scaling)
                
            elif choice == '5' and df is not None:
                method = input("Enter detection method (iqr/zscore): ") or 'iqr'
                preprocessor.detect_outliers(df, method=method)
                
            elif choice == '6' and df is not None:
                df = None
                print("Starting over...")
                
            elif (choice == '3' and df is None) or (choice == '7' and df is not None):
                print("Thank you for using Data Preprocessing Demo!")
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
    print("Data Preprocessing - Quick Demo")
    print("=" * 35)
    
    # Run complete pipeline demonstration
    original_df, processed_df = demonstrate_preprocessing_pipeline()
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_preprocessing_demo() 