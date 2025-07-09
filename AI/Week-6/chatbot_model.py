"""
Chatbot Model Implementation

This module implements the core machine learning model for intent classification
in the chatbot system. It includes neural network training, evaluation, and
prediction capabilities for natural language understanding.

Features:
- Neural network model for intent classification
- Multiple model architectures (Dense, CNN, LSTM)
- Comprehensive training and evaluation
- Model saving and loading capabilities
- Performance analysis and visualization
- Hyperparameter optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import pickle
import json
import time
from data_preprocessing import ChatbotDataPreprocessor
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ChatbotModel:
    """
    Complete chatbot model implementation with training and prediction capabilities
    """
    
    def __init__(self, model_type='dense', max_features=1000):
        """
        Initialize the chatbot model
        
        Args:
            model_type: Type of model architecture ('dense', 'cnn', 'lstm')
            max_features: Maximum number of features for text vectorization
        """
        self.model_type = model_type
        self.max_features = max_features
        self.model = None
        self.history = None
        self.preprocessor = None
        self.is_trained = False
        
        # Training parameters
        self.num_classes = 0
        self.input_dim = 0
        
        # Performance metrics
        self.training_metrics = {}
        self.evaluation_metrics = {}
        
        print(f"ChatbotModel initialized with {model_type} architecture")
    
    def build_dense_model(self, input_dim, num_classes):
        """
        Build a dense neural network model
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Dense(128, input_dim=input_dim, activation='relu', name='dense_1'),
            layers.Dropout(0.5, name='dropout_1'),
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.Dropout(0.5, name='dropout_2'),
            layers.Dense(32, activation='relu', name='dense_3'),
            layers.Dropout(0.3, name='dropout_3'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self, input_dim, num_classes):
        """
        Build a CNN model for text classification
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.Conv1D(64, 3, activation='relu', name='conv1d_1'),
            layers.MaxPooling1D(2, name='maxpool_1'),
            layers.Conv1D(32, 3, activation='relu', name='conv1d_2'),
            layers.GlobalMaxPooling1D(name='global_maxpool'),
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(0.5, name='dropout_1'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_model(self, input_dim, num_classes):
        """
        Build an LSTM model for text classification
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
            layers.LSTM(64, return_sequences=True, name='lstm_1'),
            layers.Dropout(0.3, name='dropout_1'),
            layers.LSTM(32, name='lstm_2'),
            layers.Dropout(0.3, name='dropout_2'),
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(0.5, name='dropout_3'),
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_model(self, input_dim, num_classes):
        """
        Build the specified model architecture
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        print(f"\nBuilding {self.model_type} model...")
        print(f"  Input dimension: {input_dim}")
        print(f"  Number of classes: {num_classes}")
        
        if self.model_type == 'dense':
            self.model = self.build_dense_model(input_dim, num_classes)
        elif self.model_type == 'cnn':
            self.model = self.build_cnn_model(input_dim, num_classes)
        elif self.model_type == 'lstm':
            self.model = self.build_lstm_model(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=16, validation_split=0.2, verbose=1):
        """
        Train the chatbot model
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print(f"\nTraining {self.model_type} model...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {y_train.shape[1]}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
            print(f"  Validation samples: {len(X_val)}")
        else:
            validation_data = None
            print(f"  Validation split: {validation_split}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_chatbot_model_{self.model_type}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        # Store training metrics
        self.training_metrics = {
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1],
            'final_train_accuracy': self.history.history['accuracy'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1],
            'best_val_accuracy': max(self.history.history['val_accuracy'])
        }
        
        self.is_trained = True
        
        print(f"\nTraining completed!")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Epochs trained: {self.training_metrics['epochs_trained']}")
        print(f"  Best validation accuracy: {self.training_metrics['best_val_accuracy']:.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test, class_names=None):
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            class_names: List of class names for reporting
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Detailed metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro'
        )
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'predictions': y_pred,
            'true_labels': y_true,
            'prediction_probabilities': y_pred_prob
        }
        
        # Print results
        print(f"Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Macro F1-Score: {f1_macro:.4f}")
        print(f"  Micro F1-Score: {f1_micro:.4f}")
        
        # Classification report
        if class_names is not None:
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred, target_names=class_names))
        
        return self.evaluation_metrics
    
    def predict_intent(self, text, preprocessor, confidence_threshold=0.7):
        """
        Predict intent for a given text input
        
        Args:
            text: Input text
            preprocessor: Trained preprocessor instance
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess the input
        features = preprocessor.preprocess_user_input(text, feature_type='bow')
        
        # Get prediction
        prediction_prob = self.model.predict(features, verbose=0)[0]
        predicted_class = np.argmax(prediction_prob)
        confidence = prediction_prob[predicted_class]
        
        # Get intent name
        intent = preprocessor.index_to_tag[predicted_class]
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return {
                'intent': 'unknown',
                'confidence': confidence,
                'response': "I'm not sure I understand. Could you please rephrase?",
                'all_probabilities': prediction_prob
            }
        
        # Get response
        possible_responses = preprocessor.responses[intent]
        response = np.random.choice(possible_responses)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'response': response,
            'all_probabilities': prediction_prob
        }
    
    def predict_conversational_response(self, text, preprocessor, confidence_threshold=0.7):
        """
        Predict conversational response using similarity matching
        
        Args:
            text: Input text from user
            preprocessor: Trained preprocessor with dialog data
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary with response information
        """
        if not self.is_trained:
            return {
                'response': "I'm not trained yet. Please train me first!",
                'confidence': 0.0,
                'intent': 'not_trained',
                'similar_input': None
            }
        
        # Preprocess the input text
        processed_text = preprocessor.preprocess_user_input(text, 'tfidf')
        
        # Get prediction from the model
        prediction = self.model.predict(processed_text, verbose=0)
        predicted_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_idx])
        
        # Get the most similar training input and its response
        if hasattr(preprocessor, 'response_mapping') and predicted_idx in preprocessor.response_mapping:
            response = preprocessor.response_mapping[predicted_idx]
            similar_input = preprocessor.patterns[predicted_idx] if predicted_idx < len(preprocessor.patterns) else "Unknown"
        else:
            response = "I'm not sure how to respond to that."
            similar_input = None
            confidence = 0.0
        
        # Calculate similarity with top matches for better confidence
        if hasattr(preprocessor, 'tfidf_vectorizer') and preprocessor.tfidf_vectorizer:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Get TF-IDF features for all training patterns
            if hasattr(preprocessor, 'processed_patterns'):
                training_features = preprocessor.tfidf_vectorizer.transform(preprocessor.processed_patterns)
                input_features = preprocessor.tfidf_vectorizer.transform([' '.join(preprocessor.tokenize_and_stem(preprocessor.clean_text(text)))])
                
                # Calculate similarities
                similarities = cosine_similarity(input_features, training_features).flatten()
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity > confidence:
                    confidence = float(best_similarity)
                    predicted_idx = best_match_idx
                    
                    if best_match_idx < len(preprocessor.response_mapping):
                        response = preprocessor.response_mapping[best_match_idx]
                        similar_input = preprocessor.patterns[best_match_idx]
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            return {
                'response': "I'm not sure about that. Could you rephrase your question?",
                'confidence': confidence,
                'intent': 'low_confidence',
                'similar_input': similar_input,
                'all_probabilities': prediction[0].tolist() if len(prediction[0]) <= 20 else None
            }
        
        return {
            'response': response,
            'confidence': confidence,
            'intent': 'conversational',
            'similar_input': similar_input,
            'all_probabilities': prediction[0].tolist() if len(prediction[0]) <= 20 else None
        }
    
    def visualize_training_history(self):
        """
        Visualize training history
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_type.upper()} Model Training History', fontsize=16, fontweight='bold')
        
        # Training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training and validation accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        plt.subplot(2, 2, 3)
        if 'lr' in self.history.history:
            plt.plot(self.history.history['lr'], linewidth=2)
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nHistory Not Available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate')
        
        # Training summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        summary_text = f"""
        Training Summary:
        
        Model Type: {self.model_type.upper()}
        Training Time: {self.training_metrics['training_time']:.2f}s
        Epochs Trained: {self.training_metrics['epochs_trained']}
        
        Final Results:
        Train Accuracy: {self.training_metrics['final_train_accuracy']:.4f}
        Val Accuracy: {self.training_metrics['final_val_accuracy']:.4f}
        Best Val Accuracy: {self.training_metrics['best_val_accuracy']:.4f}
        
        Overfitting: {self.training_metrics['final_train_accuracy'] - self.training_metrics['final_val_accuracy']:.4f}
        """
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_evaluation_results(self, class_names=None):
        """
        Visualize evaluation results
        """
        if not self.evaluation_metrics:
            raise ValueError("No evaluation metrics available. Call evaluate() first.")
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(self.num_classes)]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_type.upper()} Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(self.evaluation_metrics['true_labels'], 
                            self.evaluation_metrics['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Per-class metrics
        plt.subplot(2, 2, 2)
        x_pos = np.arange(len(class_names))
        width = 0.25
        
        plt.bar(x_pos - width, self.evaluation_metrics['per_class_precision'], 
               width, label='Precision', alpha=0.8)
        plt.bar(x_pos, self.evaluation_metrics['per_class_recall'], 
               width, label='Recall', alpha=0.8)
        plt.bar(x_pos + width, self.evaluation_metrics['per_class_f1'], 
               width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title('Per-Class Metrics')
        plt.xticks(x_pos, class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Prediction confidence distribution
        plt.subplot(2, 2, 3)
        max_probs = np.max(self.evaluation_metrics['prediction_probabilities'], axis=1)
        correct_mask = (self.evaluation_metrics['predictions'] == 
                       self.evaluation_metrics['true_labels'])
        
        plt.hist(max_probs[correct_mask], bins=20, alpha=0.7, 
                label='Correct Predictions', color='green')
        plt.hist(max_probs[~correct_mask], bins=20, alpha=0.7, 
                label='Incorrect Predictions', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Model performance summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        metrics_text = f"""
        Performance Summary:
        
        Overall Metrics:
        Test Accuracy: {self.evaluation_metrics['test_accuracy']:.4f}
        Test Loss: {self.evaluation_metrics['test_loss']:.4f}
        
        Macro Averages:
        Precision: {self.evaluation_metrics['precision_macro']:.4f}
        Recall: {self.evaluation_metrics['recall_macro']:.4f}
        F1-Score: {self.evaluation_metrics['f1_macro']:.4f}
        
        Micro Averages:
        Precision: {self.evaluation_metrics['precision_micro']:.4f}
        Recall: {self.evaluation_metrics['recall_micro']:.4f}
        F1-Score: {self.evaluation_metrics['f1_micro']:.4f}
        """
        
        plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model and metadata
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        # Save the Keras model
        self.model.save(f"{filepath}_model.h5")
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'max_features': self.max_features,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'training_metrics': self.training_metrics,
            'evaluation_metrics': self.evaluation_metrics
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {filepath}_model.h5")
        print(f"Metadata saved to {filepath}_metadata.pkl")
    
    def load_model(self, filepath):
        """
        Load a saved model and metadata
        
        Args:
            filepath: Path to the saved model
        """
        # Load the Keras model
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_type = metadata['model_type']
        self.max_features = metadata['max_features']
        self.input_dim = metadata['input_dim']
        self.num_classes = metadata['num_classes']
        self.training_metrics = metadata['training_metrics']
        self.evaluation_metrics = metadata['evaluation_metrics']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}_model.h5")
        print(f"Metadata loaded from {filepath}_metadata.pkl")

def compare_model_architectures():
    """
    Compare different model architectures on the same dataset
    """
    print("COMPARING MODEL ARCHITECTURES")
    print("=" * 40)
    
    # Load and preprocess data
    preprocessor = ChatbotDataPreprocessor(use_stemming=True, remove_stopwords=False)
    if not preprocessor.load_data('chatbot_data.json'):
        print("Failed to load data.")
        return
    
    preprocessor.preprocess_patterns()
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(feature_type='bow')
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Model architectures to compare
    architectures = ['dense', 'cnn', 'lstm']
    results = []
    
    for arch in architectures:
        print(f"\nTraining {arch.upper()} model...")
        
        # Create and train model
        model = ChatbotModel(model_type=arch)
        model.build_model(X_train.shape[1], y_train_cat.shape[1])
        
        # Train with fewer epochs for comparison
        history = model.train(X_train, y_train_cat, epochs=50, verbose=0)
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test_cat, 
                               class_names=list(preprocessor.index_to_tag.values()))
        
        # Store results
        results.append({
            'Architecture': arch.upper(),
            'Test_Accuracy': metrics['test_accuracy'],
            'Test_Loss': metrics['test_loss'],
            'F1_Macro': metrics['f1_macro'],
            'Training_Time': model.training_metrics['training_time'],
            'Epochs_Trained': model.training_metrics['epochs_trained']
        })
        
        print(f"{arch.upper()} - Accuracy: {metrics['test_accuracy']:.4f}, "
              f"F1: {metrics['f1_macro']:.4f}, "
              f"Time: {model.training_metrics['training_time']:.2f}s")
    
    # Display comparison
    print("\nMODEL ARCHITECTURE COMPARISON:")
    print("=" * 50)
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    return df_results

def demonstrate_chatbot_model():
    """
    Demonstrate the complete chatbot model pipeline
    """
    print("CHATBOT MODEL DEMONSTRATION")
    print("=" * 35)
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    preprocessor = ChatbotDataPreprocessor(use_stemming=True, remove_stopwords=False)
    
    if not preprocessor.load_data('chatbot_data.json'):
        print("Failed to load data. Please ensure chatbot_data.json exists.")
        return None
    
    preprocessor.preprocess_patterns()
    X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(feature_type='bow')
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    print("2. Building and training model...")
    # Create and train model
    model = ChatbotModel(model_type='dense')
    model.build_model(X_train.shape[1], y_train_cat.shape[1])
    
    # Train the model
    history = model.train(X_train, y_train_cat, epochs=100, batch_size=16, verbose=1)
    
    print("3. Evaluating model...")
    # Evaluate the model
    metrics = model.evaluate(X_test, y_test_cat, 
                           class_names=list(preprocessor.index_to_tag.values()))
    
    print("4. Creating visualizations...")
    # Visualize results
    model.visualize_training_history()
    model.visualize_evaluation_results(class_names=list(preprocessor.index_to_tag.values()))
    
    print("5. Testing predictions...")
    # Test some predictions
    test_inputs = [
        "Hello there!",
        "How are you?",
        "What can you do?",
        "Tell me a joke",
        "Goodbye!",
        "I need help with programming",
        "What's the weather like?"
    ]
    
    print("\nTesting model predictions:")
    print("-" * 40)
    for text in test_inputs:
        prediction = model.predict_intent(text, preprocessor)
        print(f"Input: '{text}'")
        print(f"  Intent: {prediction['intent']} (confidence: {prediction['confidence']:.3f})")
        print(f"  Response: {prediction['response']}")
        print()
    
    # Save the model
    print("6. Saving model...")
    model.save_model('trained_chatbot')
    preprocessor.save_preprocessor('chatbot_preprocessor.pkl')
    
    print("Model demonstration completed successfully!")
    return model, preprocessor

if __name__ == "__main__":
    # Run model demonstration
    chatbot_model, chatbot_preprocessor = demonstrate_chatbot_model()
    
    if chatbot_model:
        print("\n" + "=" * 50)
        
        # Compare different architectures
        print("Running architecture comparison...")
        comparison_results = compare_model_architectures()
        
        print("\nChatbot model training and evaluation completed!")
        print("Ready for integration into the complete chatbot system.") 