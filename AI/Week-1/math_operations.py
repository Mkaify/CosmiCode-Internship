#!/usr/bin/env python3
"""
Basic Math Operations Program
This program performs basic mathematical operations using functions.
"""

import math

def add(a, b):
    """Addition function"""
    return a + b

def subtract(a, b):
    """Subtraction function"""
    return a - b

def multiply(a, b):
    """Multiplication function"""
    return a * b

def divide(a, b):
    """Division function with zero division check"""
    if b == 0:
        return "Error: Cannot divide by zero!"
    return a / b

def power(a, b):
    """Power function (a raised to power b)"""
    return a ** b

def square_root(a):
    """Square root function"""
    if a < 0:
        return "Error: Cannot calculate square root of negative number!"
    return math.sqrt(a)

def factorial(n):
    """Factorial function"""
    if n < 0:
        return "Error: Factorial is not defined for negative numbers!"
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def percentage(part, whole):
    """Calculate percentage"""
    if whole == 0:
        return "Error: Cannot calculate percentage with zero as whole!"
    return (part / whole) * 100

def average(numbers):
    """Calculate average of a list of numbers"""
    if not numbers:
        return "Error: Cannot calculate average of empty list!"
    return sum(numbers) / len(numbers)

def find_max(numbers):
    """Find maximum number in a list"""
    if not numbers:
        return "Error: Cannot find maximum of empty list!"
    return max(numbers)

def find_min(numbers):
    """Find minimum number in a list"""
    if not numbers:
        return "Error: Cannot find minimum of empty list!"
    return min(numbers)

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def get_number_input(prompt):
    """Get a valid number input from user"""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number!")

def get_integer_input(prompt):
    """Get a valid integer input from user"""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer!")

def get_list_input():
    """Get a list of numbers from user"""
    while True:
        try:
            numbers_str = input("Enter numbers separated by spaces: ")
            numbers = [float(x) for x in numbers_str.split()]
            if not numbers:
                print("Please enter at least one number!")
                continue
            return numbers
        except ValueError:
            print("Please enter valid numbers separated by spaces!")

def display_menu():
    """Display the calculator menu"""
    print("\n" + "=" * 50)
    print("           BASIC MATH OPERATIONS CALCULATOR")
    print("=" * 50)
    print("1.  Addition")
    print("2.  Subtraction")
    print("3.  Multiplication")
    print("4.  Division")
    print("5.  Power (a^b)")
    print("6.  Square Root")
    print("7.  Factorial")
    print("8.  Percentage")
    print("9.  Average of numbers")
    print("10. Find Maximum")
    print("11. Find Minimum")
    print("12. Prime number check")
    print("13. Exit")
    print("-" * 50)

def main():
    """Main calculator function"""
    print("Welcome to the Basic Math Operations Calculator!")
    
    while True:
        display_menu()
        
        choice = input("Enter your choice (1-13): ").strip()
        
        if choice == '1':
            # Addition
            a = get_number_input("Enter first number: ")
            b = get_number_input("Enter second number: ")
            result = add(a, b)
            print(f"Result: {a} + {b} = {result}")
        
        elif choice == '2':
            # Subtraction
            a = get_number_input("Enter first number: ")
            b = get_number_input("Enter second number: ")
            result = subtract(a, b)
            print(f"Result: {a} - {b} = {result}")
        
        elif choice == '3':
            # Multiplication
            a = get_number_input("Enter first number: ")
            b = get_number_input("Enter second number: ")
            result = multiply(a, b)
            print(f"Result: {a} × {b} = {result}")
        
        elif choice == '4':
            # Division
            a = get_number_input("Enter dividend: ")
            b = get_number_input("Enter divisor: ")
            result = divide(a, b)
            print(f"Result: {a} ÷ {b} = {result}")
        
        elif choice == '5':
            # Power
            a = get_number_input("Enter base: ")
            b = get_number_input("Enter exponent: ")
            result = power(a, b)
            print(f"Result: {a}^{b} = {result}")
        
        elif choice == '6':
            # Square Root
            a = get_number_input("Enter number: ")
            result = square_root(a)
            print(f"Result: √{a} = {result}")
        
        elif choice == '7':
            # Factorial
            n = get_integer_input("Enter a non-negative integer: ")
            result = factorial(n)
            print(f"Result: {n}! = {result}")
        
        elif choice == '8':
            # Percentage
            part = get_number_input("Enter the part: ")
            whole = get_number_input("Enter the whole: ")
            result = percentage(part, whole)
            print(f"Result: {part} is {result}% of {whole}")
        
        elif choice == '9':
            # Average
            numbers = get_list_input()
            result = average(numbers)
            print(f"Numbers: {numbers}")
            print(f"Average: {result}")
        
        elif choice == '10':
            # Maximum
            numbers = get_list_input()
            result = find_max(numbers)
            print(f"Numbers: {numbers}")
            print(f"Maximum: {result}")
        
        elif choice == '11':
            # Minimum
            numbers = get_list_input()
            result = find_min(numbers)
            print(f"Numbers: {numbers}")
            print(f"Minimum: {result}")
        
        elif choice == '12':
            # Prime check
            n = get_integer_input("Enter a positive integer: ")
            result = is_prime(n)
            if result:
                print(f"{n} is a prime number!")
            else:
                print(f"{n} is not a prime number.")
        
        elif choice == '13':
            # Exit
            print("Thank you for using the Math Operations Calculator!")
            print("Goodbye! 👋")
            break
        
        else:
            print("Invalid choice! Please select a number between 1 and 13.")
        
        # Ask if user wants to continue
        continue_choice = input("\nPress Enter to continue or 'q' to quit: ").strip().lower()
        if continue_choice == 'q':
            print("Thank you for using the Math Operations Calculator!")
            break

if __name__ == "__main__":
    main() 