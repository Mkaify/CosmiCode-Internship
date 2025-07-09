"""
Test Script for Dialog-Based Chatbot

This script demonstrates the enhanced chatbot system that uses conversational
dialog data instead of intent classification. It trains on real conversation
pairs to provide more natural responses.

Features:
- Dialog-based training from dialogs.txt
- Conversational response generation
- Similarity-based matching
- Interactive testing capabilities
"""

from chatbot_engine import ChatbotEngine
import time

def test_dialog_chatbot():
    """
    Test the dialog-based chatbot system
    """
    print("DIALOG-BASED CHATBOT TESTING")
    print("=" * 40)
    
    # Initialize chatbot engine
    chatbot = ChatbotEngine(confidence_threshold=0.6)
    
    # Train from dialog data
    print("\n1. Training from dialog data...")
    success = chatbot.train_from_dialogs('dialogs.txt')
    
    if not success:
        print("Failed to train chatbot from dialog data!")
        return None
    
    # Test with sample conversations
    print("\n2. Testing conversational responses...")
    test_conversations = [
        "hi, how are you doing?",
        "i'm fine. how about yourself?",
        "how's it going?",
        "it's an ugly day today.",
        "it's such a nice day.",
        "what school do you go to?",
        "how are you feeling today?",
        "i'm having a great day!",
        "the weather is really nice",
        "i hope it doesn't rain",
        "good luck with school",
        "thank you very much"
    ]
    
    for i, test_input in enumerate(test_conversations, 1):
        print(f"\nTest {i}: '{test_input}'")
        
        start_time = time.time()
        response_data = chatbot.get_response(test_input)
        response_time = time.time() - start_time
        
        print(f"  Response: {response_data['response']}")
        print(f"  Confidence: {response_data['confidence']:.3f}")
        print(f"  Response time: {response_time:.3f}s")
        
        # Show similar input if available
        if 'similar_input' in response_data and response_data['similar_input']:
            print(f"  Similar training input: '{response_data['similar_input']}'")
    
    print("\n3. Testing edge cases...")
    edge_cases = [
        "",  # Empty input
        "asdfghjkl qwerty",  # Nonsense
        "How do you solve quantum mechanics?",  # Complex question
        "What's the meaning of life?",  # Philosophical
        "Can you help me with calculus?"  # Specific domain
    ]
    
    for test_input in edge_cases:
        if test_input:  # Skip empty input for display
            print(f"\nEdge case: '{test_input}'")
            response_data = chatbot.get_response(test_input)
            print(f"  Response: {response_data['response']}")
            print(f"  Confidence: {response_data['confidence']:.3f}")
    
    return chatbot

def interactive_test():
    """
    Interactive testing session
    """
    print("\n4. Interactive Testing Session")
    print("=" * 40)
    
    chatbot = test_dialog_chatbot()
    if chatbot is None:
        return
    
    print("\nStarting interactive session...")
    print("Type 'quit' to exit the session")
    print("-" * 30)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Session ended. Goodbye!")
                break
            
            if not user_input:
                continue
            
            response_data = chatbot.get_response(user_input)
            print(f"Bot: {response_data['response']}")
            
            # Show debug info for low confidence
            if response_data['confidence'] < 0.7:
                print(f"     (Confidence: {response_data['confidence']:.3f})")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def analyze_training_data():
    """
    Analyze the dialog training data
    """
    print("DIALOG DATA ANALYSIS")
    print("=" * 30)
    
    from data_preprocessing import ChatbotDataPreprocessor
    
    preprocessor = ChatbotDataPreprocessor()
    if preprocessor.load_dialog_data('dialogs.txt'):
        print(f"Total dialog pairs: {len(preprocessor.dialog_pairs)}")
        print(f"Total unique patterns: {len(set(preprocessor.patterns))}")
        print(f"Average pattern length: {sum(len(p.split()) for p in preprocessor.patterns) / len(preprocessor.patterns):.1f} words")
        
        # Show sample patterns
        print("\nSample dialog pairs:")
        for i in range(min(5, len(preprocessor.dialog_pairs))):
            pair = preprocessor.dialog_pairs[i]
            print(f"  Human: {pair['input']}")
            print(f"  Bot: {pair['response']}")
            print()

if __name__ == "__main__":
    # Run analysis first
    analyze_training_data()
    
    # Test the chatbot
    chatbot = test_dialog_chatbot()
    
    if chatbot:
        # Ask if user wants interactive session
        choice = input("\nWould you like to start an interactive session? (y/n): ").lower()
        if choice in ['y', 'yes']:
            interactive_test()
        else:
            print("Testing completed!") 