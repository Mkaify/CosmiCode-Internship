"""
Week 2: Search Algorithms - Linear and Binary Search Implementation
Author: AI Learning Project
Date: 2024

This module implements linear search and binary search algorithms
with examples and performance comparisons.
"""

import time
import random

def linear_search(arr, target):
    """
    Linear Search Algorithm
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        arr: List of elements to search in
        target: Element to search for
    
    Returns:
        int: Index of target element if found, -1 otherwise
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

def binary_search(arr, target):
    """
    Binary Search Algorithm (Iterative)
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Note: Array must be sorted for binary search to work
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
    
    Returns:
        int: Index of target element if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    """
    Binary Search Algorithm (Recursive)
    Time Complexity: O(log n)
    Space Complexity: O(log n) due to recursion stack
    
    Args:
        arr: Sorted list of elements to search in
        target: Element to search for
        left: Left boundary of search range
        right: Right boundary of search range
    
    Returns:
        int: Index of target element if found, -1 otherwise
    """
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def performance_comparison():
    """
    Compare performance of linear search vs binary search
    """
    print("=== Performance Comparison: Linear vs Binary Search ===")
    
    # Generate test data
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        # Create sorted array
        arr = list(range(1, size + 1))
        target = random.randint(1, size)
        
        print(f"\nArray size: {size:,}")
        print(f"Searching for: {target}")
        
        # Linear Search
        start_time = time.time()
        linear_result = linear_search(arr, target)
        linear_time = time.time() - start_time
        
        # Binary Search
        start_time = time.time()
        binary_result = binary_search(arr, target)
        binary_time = time.time() - start_time
        
        print(f"Linear Search: Found at index {linear_result}, Time: {linear_time:.6f}s")
        print(f"Binary Search: Found at index {binary_result}, Time: {binary_time:.6f}s")
        
        if binary_time > 0:
            speedup = linear_time / binary_time
            print(f"Binary search is {speedup:.2f}x faster")

def demonstrate_searches():
    """
    Demonstrate both search algorithms with examples
    """
    print("=== Search Algorithms Demonstration ===")
    
    # Test data
    numbers = [3, 7, 12, 18, 23, 31, 45, 56, 67, 78, 89, 94]
    print(f"Sorted array: {numbers}")
    
    test_cases = [23, 45, 100, 3, 94]
    
    print("\n--- Linear Search Results ---")
    for target in test_cases:
        result = linear_search(numbers, target)
        if result != -1:
            print(f"Linear Search: Found {target} at index {result}")
        else:
            print(f"Linear Search: {target} not found")
    
    print("\n--- Binary Search Results ---")
    for target in test_cases:
        result = binary_search(numbers, target)
        if result != -1:
            print(f"Binary Search: Found {target} at index {result}")
        else:
            print(f"Binary Search: {target} not found")
    
    print("\n--- Binary Search Recursive Results ---")
    for target in test_cases:
        result = binary_search_recursive(numbers, target)
        if result != -1:
            print(f"Binary Search (Recursive): Found {target} at index {result}")
        else:
            print(f"Binary Search (Recursive): {target} not found")

def search_strings():
    """
    Demonstrate search algorithms with string data
    """
    print("\n=== String Search Example ===")
    
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]
    print(f"Names list: {names}")
    
    search_names = ["Charlie", "Frank", "Zoe"]
    
    for name in search_names:
        linear_result = linear_search(names, name)
        binary_result = binary_search(names, name)  # Works because list is sorted
        
        print(f"Searching for '{name}':")
        print(f"  Linear: {'Found at index ' + str(linear_result) if linear_result != -1 else 'Not found'}")
        print(f"  Binary: {'Found at index ' + str(binary_result) if binary_result != -1 else 'Not found'}")

if __name__ == "__main__":
    print("Week 2: Linear and Binary Search Implementation\n")
    
    # Run demonstrations
    demonstrate_searches()
    search_strings()
    performance_comparison()
    
    print("\n=== Key Takeaways ===")
    print("1. Linear Search: O(n) time, works on unsorted arrays")
    print("2. Binary Search: O(log n) time, requires sorted arrays")
    print("3. Binary search is significantly faster for large datasets")
    print("4. Binary search can be implemented iteratively or recursively") 