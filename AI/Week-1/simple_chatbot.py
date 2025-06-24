#!/usr/bin/env python3
"""
Simple Rule-Based Chatbot
This chatbot responds to user input using if-else conditions.
"""

import random
import datetime

def get_response(user_input):
    """Generate response based on user input using if-else conditions"""
    
    # Convert input to lowercase for easier matching
    user_input = user_input.lower().strip()
    
    # Greeting responses
    if any(greeting in user_input for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        responses = [
            "Hello! How can I help you today?",
            "Hi there! What would you like to know?",
            "Hey! I'm here to assist you.",
            "Greetings! How may I assist you?"
        ]
        return random.choice(responses)
    
    # How are you responses
    elif any(phrase in user_input for phrase in ['how are you', 'how do you do', 'what\'s up']):
        responses = [
            "I'm doing great! Thanks for asking. How are you?",
            "I'm functioning perfectly! How about you?",
            "All systems running smoothly! How can I help you today?"
        ]
        return random.choice(responses)
    
    # Name related questions
    elif any(phrase in user_input for phrase in ['what is your name', 'your name', 'who are you']):
        return "I'm ChatBot, your friendly AI assistant! What's your name?"
    
    # Time related questions
    elif any(phrase in user_input for phrase in ['what time', 'current time', 'time now']):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}"
    
    # Date related questions
    elif any(phrase in user_input for phrase in ['what date', 'today\'s date', 'current date']):
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        return f"Today's date is {current_date}"
    
    # AI related questions
    elif any(phrase in user_input for phrase in ['what is ai', 'artificial intelligence', 'machine learning']):
        return "AI (Artificial Intelligence) is the simulation of human intelligence in machines. It includes machine learning, neural networks, and various algorithms that help computers learn and make decisions!"
    
    # Programming related questions
    elif any(phrase in user_input for phrase in ['python', 'programming', 'coding']):
        return "Python is a fantastic programming language for AI and machine learning! It's beginner-friendly and has powerful libraries like NumPy, Pandas, and TensorFlow."
    
    # Weather (mock response since we don't have real weather data)
    elif any(phrase in user_input for phrase in ['weather', 'temperature', 'climate']):
        return "I don't have access to real-time weather data, but I recommend checking your local weather app or website for accurate information!"
    
    # Help requests
    elif any(phrase in user_input for phrase in ['help', 'assist', 'support']):
        return "I'm here to help! I can chat about AI, programming, tell you the time/date, or just have a friendly conversation. What would you like to know?"
    
    # Math operations
    elif any(phrase in user_input for phrase in ['calculate', 'math', 'add', 'subtract', 'multiply', 'divide']):
        return "I can help with basic math! Try asking me something like 'what is 5 + 3?' or 'calculate 10 * 7'"
    
    # Simple math calculations
    elif 'what is' in user_input and any(op in user_input for op in ['+', '-', '*', '/', 'plus', 'minus', 'times', 'divided']):
        return calculate_math(user_input)
    
    # Goodbye responses
    elif any(phrase in user_input for phrase in ['bye', 'goodbye', 'see you', 'exit', 'quit']):
        responses = [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! It was nice chatting with you!",
            "Until next time! Have a wonderful day!"
        ]
        return random.choice(responses)
    
    # Thank you responses
    elif any(phrase in user_input for phrase in ['thank you', 'thanks', 'appreciate']):
        responses = [
            "You're welcome! Happy to help!",
            "No problem! Anything else I can help with?",
            "Glad I could help! Is there anything else?"
        ]
        return random.choice(responses)
    
    # Default responses for unrecognized input
    else:
        responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "That's interesting! Can you tell me more?",
            "I'm still learning. Could you try asking something else?",
            "Hmm, I don't quite get that. Try asking about AI, programming, or the time!",
            "I'm a simple chatbot. Try greeting me or asking about AI!"
        ]
        return random.choice(responses)

def calculate_math(user_input):
    """Simple math calculator for basic operations"""
    try:
        # Replace words with symbols
        user_input = user_input.replace('plus', '+').replace('add', '+')
        user_input = user_input.replace('minus', '-').replace('subtract', '-')
        user_input = user_input.replace('times', '*').replace('multiply', '*')
        user_input = user_input.replace('divided by', '/').replace('divide', '/')
        
        # Extract the mathematical expression
        import re
        math_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)'
        match = re.search(math_pattern, user_input)
        
        if match:
            num1, operator, num2 = match.groups()
            num1, num2 = float(num1), float(num2)
            
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                if num2 != 0:
                    result = num1 / num2
                else:
                    return "Sorry, I can't divide by zero!"
            
            return f"{num1} {operator} {num2} = {result}"
        else:
            return "I couldn't understand the math problem. Try something like '5 + 3' or 'what is 10 * 2?'"
    
    except:
        return "Sorry, I had trouble with that calculation. Try a simpler format like '5 + 3'"

def main():
    """Main chatbot loop"""
    print("🤖 ChatBot: Hello! I'm a simple chatbot. Type 'bye' to exit.")
    print("You can ask me about AI, programming, time, date, or just chat!")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit conditions
        if user_input.lower() in ['bye', 'goodbye', 'exit', 'quit']:
            print("🤖 ChatBot: Goodbye! Have a great day! 👋")
            break
        
        # Skip empty input
        if not user_input:
            print("🤖 ChatBot: Please say something!")
            continue
        
        # Get and display response
        response = get_response(user_input)
        print(f"🤖 ChatBot: {response}")

if __name__ == "__main__":
    main() 