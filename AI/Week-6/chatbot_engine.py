"""
Complete Chatbot Engine

This module integrates all components to create a fully functional chatbot system.
It includes conversation management, response generation, context handling,
and an interactive user interface.

Features:
- Complete conversation engine
- Context-aware responses
- Confidence-based response selection
- Interactive console interface
- Conversation logging and analysis
- Fallback mechanisms for unknown inputs
- Session management
"""

import numpy as np
import pandas as pd
import json
import pickle
import datetime
import time
import random
from collections import defaultdict, deque
import re
from tensorflow import keras
from data_preprocessing import ChatbotDataPreprocessor
from chatbot_model import ChatbotModel
import warnings
warnings.filterwarnings('ignore')

class ChatbotEngine:
    """
    Complete chatbot engine with conversation management
    """
    
    def __init__(self, model_path=None, preprocessor_path=None, confidence_threshold=0.7):
        """
        Initialize the chatbot engine
        
        Args:
            model_path: Path to saved model (without extension)
            preprocessor_path: Path to saved preprocessor
            confidence_threshold: Minimum confidence for predictions
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.preprocessor = None
        
        # Conversation state
        self.conversation_history = deque(maxlen=10)  # Keep last 10 exchanges
        self.session_start_time = None
        self.total_interactions = 0
        self.unknown_inputs = []
        
        # Context tracking
        self.current_context = None
        self.context_stack = []
        
        # Statistics
        self.intent_counts = defaultdict(int)
        self.response_times = []
        self.confidence_scores = []
        
        # Load model and preprocessor if paths provided
        if model_path and preprocessor_path:
            self.load_components(model_path, preprocessor_path)
        
        # Fallback responses for low confidence predictions
        self.fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "That's interesting! Could you tell me more about what you're looking for?",
            "I'm still learning. Can you try asking that in a different way?",
            "I didn't quite catch that. Could you be more specific?",
            "Hmm, I'm not sure about that. Is there something else I can help you with?",
            "I'm having trouble understanding. Could you try asking differently?",
            "That's a bit unclear to me. Can you give me more details?"
        ]
        
        print("ChatbotEngine initialized!")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def load_components(self, model_path, preprocessor_path):
        """
        Load trained model and preprocessor
        
        Args:
            model_path: Path to saved model (without extension)
            preprocessor_path: Path to saved preprocessor
        """
        import os
        try:
            print("Loading chatbot components...")
            
            # Handle relative path resolution for preprocessor
            if not os.path.isabs(preprocessor_path) and not os.path.exists(preprocessor_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                potential_path = os.path.join(script_dir, preprocessor_path)
                if os.path.exists(potential_path):
                    preprocessor_path = potential_path
            
            # Handle relative path resolution for model (check for .h5 extension)
            model_file = f"{model_path}.h5"
            if not os.path.isabs(model_file) and not os.path.exists(model_file):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                potential_path = os.path.join(script_dir, model_file)
                if os.path.exists(potential_path):
                    model_path = os.path.join(script_dir, model_path)
            
            # Load preprocessor
            self.preprocessor = ChatbotDataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
            
            # Load model
            self.model = ChatbotModel()
            self.model.load_model(model_path)
            
            print("✓ Components loaded successfully!")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            print("Starting with training mode...")
            return False
        
        return True
    

    
    def train_from_dialogs(self, dialog_file='dialogs.txt'):
        """
        Train the chatbot from dialog data file
        
        Args:
            dialog_file: Path to dialog training data file
        """
        import os
        print("Training chatbot from dialog data...")
        
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
                if os.path.exists(dialog_file):
                    print(f"Found {dialog_file} in current directory")
                elif os.path.exists(potential_path):
                    dialog_file = potential_path
                    print(f"Found {dialog_file} in script directory")
                else:
                    print(f"Could not find {dialog_file} in either location")
        
        # Initialize and load dialog data
        self.preprocessor = ChatbotDataPreprocessor(use_stemming=True, remove_stopwords=False, max_features=2000)
        
        if not self.preprocessor.load_dialog_data(dialog_file):
            print(f"Failed to load dialog data from {dialog_file}")
            return False
        
        # Prepare training data for conversational model
        training_data = self.preprocessor.prepare_conversational_data(feature_type='tfidf')
        if training_data is None:
            print("Failed to prepare training data")
            return False
        
        X_train, X_test, y_train, y_test = training_data
        
        # Convert to categorical for the number of response options
        from tensorflow.keras.utils import to_categorical
        num_responses = len(self.preprocessor.response_mapping)
        y_train_cat = to_categorical(y_train, num_classes=num_responses)
        y_test_cat = to_categorical(y_test, num_classes=num_responses)
        
        # Create and train model
        self.model = ChatbotModel(model_type='dense')
        self.model.build_model(X_train.shape[1], num_responses)
        
        # Train the model
        print(f"\nStarting training with {len(self.preprocessor.dialog_pairs)} dialog pairs...")
        self.model.train(X_train, y_train_cat, epochs=50, batch_size=32, verbose=1)
        
        # Evaluate
        test_loss, test_accuracy = self.model.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Training completed! Test accuracy: {test_accuracy:.4f}")
        
        # Save components
        self.save_components()
        
        return True
    
    def save_components(self, model_path='trained_dialog_chatbot', preprocessor_path='dialog_preprocessor.pkl'):
        """
        Save trained components
        
        Args:
            model_path: Path to save model (without extension)
            preprocessor_path: Path to save preprocessor
        """
        import os
        if self.model and self.preprocessor:
            # Save in the same directory as the script if using relative paths
            if not os.path.isabs(model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, model_path)
            
            if not os.path.isabs(preprocessor_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                preprocessor_path = os.path.join(script_dir, preprocessor_path)
            
            self.model.save_model(model_path)
            self.preprocessor.save_preprocessor(preprocessor_path)
            print(f"Components saved successfully!")
            print(f"Model saved to: {model_path}")
            print(f"Preprocessor saved to: {preprocessor_path}")
        else:
            print("No trained components to save!")
    
    def preprocess_input(self, user_input):
        """
        Preprocess user input
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Cleaned input string
        """
        # Basic input cleaning
        cleaned = user_input.strip()
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Handle empty input
        if not cleaned:
            return None
        
        return cleaned
    
    def get_response(self, user_input):
        """
        Generate response for user input
        
        Args:
            user_input: User's message
            
        Returns:
            Dictionary with response information
        """
        start_time = time.time()
        
        # Preprocess input
        cleaned_input = self.preprocess_input(user_input)
        if cleaned_input is None:
            return {
                'response': "I didn't receive any input. Could you try again?",
                'intent': 'empty_input',
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
        
        # Check if model is available
        if self.model is None or self.preprocessor is None:
            return {
                'response': "I'm still learning! Please train me first or load a trained model.",
                'intent': 'not_trained',
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
        
        # Get prediction
        try:
            # Use dialog-based conversational prediction
            prediction = self.model.predict_conversational_response(cleaned_input, self.preprocessor, 
                                                                  self.confidence_threshold)
            
            # Process prediction
            intent = prediction['intent']
            confidence = prediction['confidence']
            response = prediction['response']
            
            # Handle low confidence
            if intent in ['unknown', 'low_confidence'] or confidence < self.confidence_threshold:
                response = random.choice(self.fallback_responses)
                intent = 'unknown'
                self.unknown_inputs.append(cleaned_input)
            
            # Update statistics
            self.intent_counts[intent] += 1
            self.confidence_scores.append(confidence)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            response = "I'm having some technical difficulties. Please try again."
            intent = 'error'
            confidence = 0.0
        
        processing_time = time.time() - start_time
        self.response_times.append(processing_time)
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'processing_time': processing_time,
            'all_probabilities': prediction.get('all_probabilities', None)
        }
    
    def start_conversation(self):
        """
        Start a new conversation session
        """
        self.session_start_time = datetime.datetime.now()
        self.conversation_history.clear()
        self.total_interactions = 0
        
        # Welcome message
        welcome_messages = [
            "Hello! I'm your AI chatbot assistant. How can I help you today?",
            "Hi there! I'm here to chat and assist you. What would you like to talk about?",
            "Welcome! I'm an AI chatbot ready to help with your questions. What's on your mind?",
            "Greetings! I'm your friendly AI assistant. How may I be of service today?",
            "Hello! I'm an AI-powered chatbot. Feel free to ask me anything or just chat!"
        ]
        
        print("\n" + "="*60)
        print("🤖 CHATBOT CONVERSATION STARTED")
        print("="*60)
        print(random.choice(welcome_messages))
        print("\nType 'quit', 'exit', or 'bye' to end the conversation.")
        print("Type 'help' for available commands.")
        print("Type 'stats' to see conversation statistics.")
        print("-"*60)
    
    def end_conversation(self):
        """
        End the conversation and show statistics
        """
        if self.session_start_time:
            session_duration = datetime.datetime.now() - self.session_start_time
            
            print("\n" + "="*60)
            print("🤖 CONVERSATION ENDED")
            print("="*60)
            
            # Session statistics
            print("Session Statistics:")
            print(f"  Duration: {session_duration}")
            print(f"  Total interactions: {self.total_interactions}")
            
            if self.response_times:
                print(f"  Average response time: {np.mean(self.response_times):.3f} seconds")
            
            if self.confidence_scores:
                print(f"  Average confidence: {np.mean(self.confidence_scores):.3f}")
            
            # Intent breakdown
            if self.intent_counts:
                print("\nIntent Distribution:")
                for intent, count in sorted(self.intent_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {intent}: {count}")
            
            # Unknown inputs
            if self.unknown_inputs:
                print(f"\nUnknown inputs encountered: {len(self.unknown_inputs)}")
                print("Examples:")
                for example in self.unknown_inputs[:3]:
                    print(f"  - '{example}'")
            
            print("\nThank you for chatting! Have a great day! 👋")
            print("="*60)
    
    def show_help(self):
        """
        Show help information
        """
        help_text = """
Available Commands:
  quit, exit, bye    - End the conversation
  help              - Show this help message
  stats             - Show current session statistics
  clear             - Clear conversation history
  confidence        - Show confidence threshold settings
  
Tips for better conversations:
  • Ask clear, specific questions
  • Use natural language
  • Try different phrasings if I don't understand
  • Be patient - I'm still learning!
  
I can help with:
  • General questions and information
  • Technology and programming topics
  • Casual conversation and small talk
  • Jokes and entertainment
  • Learning and educational topics
        """
        print(help_text)
    
    def show_stats(self):
        """
        Show current session statistics
        """
        if self.session_start_time:
            duration = datetime.datetime.now() - self.session_start_time
            print(f"\nSession Statistics:")
            print(f"  Duration: {duration}")
            print(f"  Interactions: {self.total_interactions}")
            
            if self.response_times:
                print(f"  Avg response time: {np.mean(self.response_times):.3f}s")
            
            if self.confidence_scores:
                print(f"  Avg confidence: {np.mean(self.confidence_scores):.3f}")
                print(f"  Min confidence: {min(self.confidence_scores):.3f}")
                print(f"  Max confidence: {max(self.confidence_scores):.3f}")
            
            print(f"  Unknown inputs: {len(self.unknown_inputs)}")
        else:
            print("No active session statistics available.")
    
    def interactive_chat(self):
        """
        Run interactive chat session
        """
        self.start_conversation()
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Handle empty input
                if not user_input:
                    print("🤖 Bot: I didn't receive any input. Please try again.")
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    # Get a farewell response first
                    if self.model and self.preprocessor:
                        farewell_response = self.get_response(user_input)
                        print(f"🤖 Bot: {farewell_response['response']}")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🤖 Bot: Conversation history cleared!")
                    continue
                
                elif user_input.lower() == 'confidence':
                    print(f"🤖 Bot: Current confidence threshold: {self.confidence_threshold}")
                    continue
                
                # Get bot response
                response_data = self.get_response(user_input)
                
                # Display response
                bot_response = response_data['response']
                intent = response_data['intent']
                confidence = response_data['confidence']
                
                print(f"🤖 Bot: {bot_response}")
                
                # Show debug info for unknown intents
                if intent == 'unknown':
                    print(f"    💭 (I'm not confident about this response - confidence: {confidence:.3f})")
                
                # Store conversation
                self.conversation_history.append({
                    'user_input': user_input,
                    'bot_response': bot_response,
                    'intent': intent,
                    'confidence': confidence,
                    'timestamp': datetime.datetime.now()
                })
                
                self.total_interactions += 1
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bot: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"🤖 Bot: Sorry, I encountered an error: {e}")
                print("🤖 Bot: Let's try again!")
        
        self.end_conversation()
    
    def batch_test(self, test_inputs):
        """
        Test the chatbot with a batch of inputs
        
        Args:
            test_inputs: List of test input strings
            
        Returns:
            List of response dictionaries
        """
        print("Running batch test...")
        results = []
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\nTest {i}/{len(test_inputs)}: '{test_input}'")
            
            response_data = self.get_response(test_input)
            results.append({
                'input': test_input,
                'response': response_data['response'],
                'intent': response_data['intent'],
                'confidence': response_data['confidence']
            })
            
            print(f"  Intent: {response_data['intent']}")
            print(f"  Confidence: {response_data['confidence']:.3f}")
            print(f"  Response: {response_data['response']}")
        
        return results
    
    def export_conversation_log(self, filename=None):
        """
        Export conversation history to file
        
        Args:
            filename: Output filename (optional)
        """
        if not self.conversation_history:
            print("No conversation history to export.")
            return
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chatbot_conversation_{timestamp}.json"
        
        # Prepare data for export
        conversation_data = {
            'session_info': {
                'start_time': self.session_start_time.isoformat() if self.session_start_time else None,
                'total_interactions': self.total_interactions,
                'confidence_threshold': self.confidence_threshold
            },
            'conversation_history': [
                {
                    'user_input': exchange['user_input'],
                    'bot_response': exchange['bot_response'],
                    'intent': exchange['intent'],
                    'confidence': exchange['confidence'],
                    'timestamp': exchange['timestamp'].isoformat()
                }
                for exchange in self.conversation_history
            ],
            'statistics': {
                'intent_counts': dict(self.intent_counts),
                'avg_response_time': np.mean(self.response_times) if self.response_times else 0,
                'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
                'unknown_inputs': self.unknown_inputs
            }
        }
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        print(f"Conversation log exported to {filename}")

def demo_chatbot():
    """
    Demonstration of the complete chatbot system
    """
    print("COMPLETE CHATBOT SYSTEM DEMONSTRATION")
    print("=" * 45)
    
    # Initialize chatbot
    chatbot = ChatbotEngine(confidence_threshold=0.7)
    
    # Check if trained dialog model exists
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        success = chatbot.load_components('trained_dialog_chatbot', 'dialog_preprocessor.pkl')
        if not success:
            raise FileNotFoundError("No trained dialog components found")
        print("✓ Loaded existing dialog-based chatbot model")
    except:
        print("No pre-trained model found. Training from dialog data...")
        
        success = chatbot.train_from_dialogs('dialogs.txt')
        
        if not success:
            print("Failed to train chatbot from dialog data.")
            print("Please ensure dialogs.txt exists in the script directory.")
            expected_dialog_path = os.path.join(script_dir, 'dialogs.txt')
            print(f"Expected dialog file: {expected_dialog_path}")
            return None
        else:
            print("✓ Successfully trained new dialog-based chatbot model")
    
    # Run batch tests with conversational examples
    test_inputs = [
        "hi, how are you doing?",
        "i'm fine. how about yourself?",
        "how's it going?",
        "it's an ugly day today.",
        "it's such a nice day.",
        "what school do you go to?",
        "good luck with school.",
        "thank you very much",
        "i really want to go to the beach this weekend.",
        "hello, may i speak to alice please?"
    ]
    
    print("\n" + "="*50)
    print("BATCH TESTING")
    print("="*50)
    
    batch_results = chatbot.batch_test(test_inputs)
    
    # Analyze batch results
    print("\nBatch Test Analysis:")
    high_confidence = [r for r in batch_results if r['confidence'] > 0.8]
    low_confidence = [r for r in batch_results if r['confidence'] < 0.5]
    
    print(f"  High confidence responses (>0.8): {len(high_confidence)}/{len(batch_results)}")
    print(f"  Low confidence responses (<0.5): {len(low_confidence)}/{len(batch_results)}")
    
    if low_confidence:
        print("  Low confidence examples:")
        for result in low_confidence[:3]:
            print(f"    '{result['input']}' -> {result['confidence']:.3f}")
    
    return chatbot

def main():
    """
    Main function to run the chatbot
    """
    # Create and test chatbot
    chatbot = demo_chatbot()
    
    if chatbot is None:
        print("Failed to initialize chatbot.")
        return
    
    # Ask user if they want to start interactive chat
    print("\n" + "="*50)
    choice = input("Would you like to start an interactive chat session? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        chatbot.interactive_chat()
    else:
        print("Demo completed. You can start interactive chat anytime by calling chatbot.interactive_chat()")

if __name__ == "__main__":
    main() 