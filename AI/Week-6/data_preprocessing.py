"""
Data Preprocessing for Dialog-Based Chatbot

This module handles data preprocessing for conversational dialog training:
- Loading and parsing dialog data from text files
- Text cleaning and normalization
- Tokenization and stemming
- Feature extraction (TF-IDF)
- Data preparation for conversational model training

Features:
- Dialog pair preprocessing pipeline
- TF-IDF feature extraction
- Similarity-based response matching
- Clean and efficient processing
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import os
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Checking and downloading required NLTK data...")

# Check for punkt_tab (newer version)
try:
    nltk.data.find('tokenizers/punkt_tab')
    print("✓ punkt_tab tokenizer found")
except LookupError:
    print("Downloading punkt_tab tokenizer...")
    try:
        nltk.download('punkt_tab', quiet=True)
        print("✓ punkt_tab downloaded successfully")
    except Exception as e:
        print(f"Failed to download punkt_tab: {e}")
        print("Trying punkt (older version)...")
        try:
            nltk.download('punkt', quiet=True)
            print("✓ punkt downloaded as fallback")
        except Exception as e2:
            print(f"Failed to download punkt: {e2}")

# Check for stopwords
try:
    nltk.data.find('corpora/stopwords')
    print("✓ stopwords corpus found")
except LookupError:
    print("Downloading stopwords corpus...")
    try:
        nltk.download('stopwords', quiet=True)
        print("✓ stopwords downloaded successfully")
    except Exception as e:
        print(f"Failed to download stopwords: {e}")

print("NLTK data check completed!")

# Test NLTK functionality
try:
    test_tokens = word_tokenize("This is a test sentence.")
    print(f"✓ NLTK tokenization working: {test_tokens}")
except Exception as e:
    print(f"⚠ NLTK tokenization test failed: {e}")

class ChatbotDataPreprocessor:
    """
    Dialog-based data preprocessing for conversational chatbot training
    """
    
    def __init__(self, use_stemming=True, remove_stopwords=False, max_features=2000):
        """
        Initialize the preprocessor for dialog data
        
        Args:
            use_stemming: Whether to apply stemming
            remove_stopwords: Whether to remove stop words
            max_features: Maximum number of features for vectorization
        """
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.max_features = max_features
        
        # Initialize NLTK components
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Dialog data storage
        self.dialog_pairs = []
        self.patterns = []
        self.responses_list = []
        self.response_mapping = {}
        
        # Processed data
        self.processed_patterns = []
        self.vocabulary = set()
        
        # Feature extractors
        self.tfidf_vectorizer = None
        self.bow_vectorizer = None
        
        print("Dialog-based ChatbotDataPreprocessor initialized!")
        print(f"  Stemming: {use_stemming}")
        print(f"  Remove stopwords: {remove_stopwords}")
        print(f"  Max features: {max_features}")
    

    
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_stem(self, text):
        """
        Tokenize text and optionally apply stemming
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if specified
        if self.use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_patterns(self):
        """
        Preprocess all patterns in the dataset
        """
        print("\nPreprocessing patterns...")
        
        self.processed_patterns = []
        all_tokens = []
        
        for pattern in self.patterns:
            # Clean text
            cleaned = self.clean_text(pattern)
            
            # Tokenize and stem
            tokens = self.tokenize_and_stem(cleaned)
            
            # Store processed pattern
            processed_pattern = ' '.join(tokens)
            self.processed_patterns.append(processed_pattern)
            
            # Collect all tokens for vocabulary
            all_tokens.extend(tokens)
        
        # Build vocabulary
        self.vocabulary = set(all_tokens)
        
        print(f"Preprocessing completed!")
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        print(f"  Average tokens per pattern: {len(all_tokens) / len(self.patterns):.2f}")
    

    
    def create_tfidf_features(self):
        """
        Create TF-IDF feature vectors
        
        Returns:
            Feature matrix and vectorizer
        """
        print("\nCreating TF-IDF features...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform the processed patterns
        tfidf_features = self.tfidf_vectorizer.fit_transform(self.processed_patterns)
        
        print(f"TF-IDF features created!")
        print(f"  Feature matrix shape: {tfidf_features.shape}")
        print(f"  Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        return tfidf_features.toarray(), self.tfidf_vectorizer
    
    def create_bag_of_words_features(self):
        """
        Create Bag of Words feature vectors
        
        Returns:
            Feature matrix and vectorizer
        """
        print("\nCreating Bag of Words features...")
        
        from sklearn.feature_extraction.text import CountVectorizer
        
        self.bow_vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Fit and transform the processed patterns
        bow_features = self.bow_vectorizer.fit_transform(self.processed_patterns)
        
        print(f"Bag of Words features created!")
        print(f"  Feature matrix shape: {bow_features.shape}")
        print(f"  Vocabulary size: {len(self.bow_vectorizer.vocabulary_)}")
        
        return bow_features.toarray(), self.bow_vectorizer
    
    def save_preprocessor(self, filepath):
        """
        Save the preprocessor and its components
        
        Args:
            filepath: Path to save the preprocessor
        """
        preprocessor_data = {
            'dialog_pairs': self.dialog_pairs,
            'patterns': self.patterns,
            'responses_list': self.responses_list,
            'response_mapping': self.response_mapping,
            'processed_patterns': self.processed_patterns,
            'vocabulary': list(self.vocabulary),
            'use_stemming': self.use_stemming,
            'remove_stopwords': self.remove_stopwords,
            'max_features': self.max_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
        
        # Save vectorizers separately if they exist
        if hasattr(self, 'bow_vectorizer') and self.bow_vectorizer:
            with open(filepath.replace('.pkl', '_bow_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.bow_vectorizer, f)
        
        if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer:
            with open(filepath.replace('.pkl', '_tfidf_vectorizer.pkl'), 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Load a saved dialog preprocessor
        
        Args:
            filepath: Path to the saved preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.dialog_pairs = preprocessor_data.get('dialog_pairs', [])
        self.patterns = preprocessor_data.get('patterns', [])
        self.responses_list = preprocessor_data.get('responses_list', [])
        self.response_mapping = preprocessor_data.get('response_mapping', {})
        self.processed_patterns = preprocessor_data.get('processed_patterns', [])
        self.vocabulary = set(preprocessor_data.get('vocabulary', []))
        self.use_stemming = preprocessor_data.get('use_stemming', True)
        self.remove_stopwords = preprocessor_data.get('remove_stopwords', False)
        self.max_features = preprocessor_data.get('max_features', 2000)
        
        # Load vectorizers if they exist
        tfidf_path = filepath.replace('.pkl', '_tfidf_vectorizer.pkl')
        try:
            with open(tfidf_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
        except FileNotFoundError:
            pass
        
        bow_path = filepath.replace('.pkl', '_bow_vectorizer.pkl')
        try:
            with open(bow_path, 'rb') as f:
                self.bow_vectorizer = pickle.load(f)
        except FileNotFoundError:
            pass
        
        print(f"Dialog preprocessor loaded from {filepath}")
        print(f"  Loaded {len(self.dialog_pairs)} dialog pairs")
    
    def preprocess_user_input(self, text, feature_type='tfidf'):
        """
        Preprocess user input for prediction using TF-IDF
        
        Args:
            text: User input text
            feature_type: Type of features to use (only 'tfidf' supported)
            
        Returns:
            Feature vector for the input text
        """
        # Clean and preprocess the text
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_stem(cleaned)
        processed_text = ' '.join(tokens)
        
        # Transform using TF-IDF vectorizer
        if self.tfidf_vectorizer:
            features = self.tfidf_vectorizer.transform([processed_text])
            return features.toarray()
        else:
            raise ValueError("TF-IDF vectorizer not available! Train the model first.")
    
    def load_dialog_data(self, dialog_file):
        """
        Load conversational dialog data from text file
        
        Args:
            dialog_file: Path to the dialog file (tab-separated pairs)
            
        Returns:
            bool: Success status
        """
        print(f"\nLoading dialog data from {dialog_file}...")
        
        # Handle relative path resolution
        if not os.path.isabs(dialog_file) and not os.path.exists(dialog_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(script_dir, dialog_file)
            if os.path.exists(potential_path):
                dialog_file = potential_path
            else:
                print(f"Searching for {dialog_file} in current directory and script directory...")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Script directory: {script_dir}")
        
        try:
            with open(dialog_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"Error: File {dialog_file} not found!")
            return False
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
        
        # Parse dialog pairs
        self.dialog_pairs = []
        self.patterns = []
        self.responses_list = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            # Split by tab
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Line {line_num} doesn't have exactly 2 tab-separated parts: {line[:50]}...")
                continue
            
            human_msg, bot_response = parts
            
            # Store the dialog pair
            self.dialog_pairs.append({
                'input': human_msg.strip(),
                'response': bot_response.strip()
            })
            
            # Store for training (input patterns)
            self.patterns.append(human_msg.strip())
            self.responses_list.append(bot_response.strip())
        
        # Create a simple mapping for responses (we'll use similarity matching)
        self.response_mapping = {i: response for i, response in enumerate(self.responses_list)}
        
        print(f"Dialog data loaded successfully!")
        print(f"  Total dialog pairs: {len(self.dialog_pairs)}")
        print(f"  Total patterns: {len(self.patterns)}")
        print(f"  Sample pairs:")
        for i in range(min(3, len(self.dialog_pairs))):
            print(f"    Human: {self.dialog_pairs[i]['input']}")
            print(f"    Bot: {self.dialog_pairs[i]['response']}")
        
        return True
    
    def prepare_conversational_data(self, feature_type='tfidf', test_size=0.2, random_state=42):
        """
        Prepare training data for conversational model
        
        Args:
            feature_type: Type of features ('bow', 'tfidf')
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of training data (X_train, X_test, y_train, y_test, response_indices)
        """
        if not hasattr(self, 'dialog_pairs') or not self.dialog_pairs:
            print("Error: No dialog data loaded. Use load_dialog_data() first.")
            return None
        
        print(f"\nPreparing conversational training data...")
        print(f"Feature type: {feature_type}")
        
        # Preprocess patterns
        self.preprocess_patterns()
        
        # Create features
        if feature_type == 'bow':
            X, _ = self.create_bag_of_words_features()
        elif feature_type == 'tfidf':
            X, _ = self.create_tfidf_features()
        else:
            raise ValueError("feature_type must be 'bow' or 'tfidf'")
        
        # For conversational data, y represents the index of the corresponding response
        y = np.arange(len(self.patterns))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Training data prepared:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Feature dimensions: {X_train.shape[1]}")
        print(f"  Response vocabulary size: {len(self.response_mapping)}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Dialog-based Data Preprocessing Module")
    print("Use this module with the ChatbotEngine for dialog training.") 