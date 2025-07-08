"""
Towers of Hanoi Problem Solver using Recursion

The Towers of Hanoi is a classic problem involving three pegs and n disks of different sizes.
The goal is to move all disks from the source peg to the destination peg following these rules:
1. Only one disk can be moved at a time
2. Only the top disk can be moved
3. A larger disk cannot be placed on top of a smaller disk

Time Complexity: O(2^n)
Space Complexity: O(n) due to recursion stack
"""

def towers_of_hanoi(n, source, destination, auxiliary, moves_list=None):
    """
    Solve Towers of Hanoi using recursion
    
    Args:
        n: Number of disks
        source: Source peg identifier
        destination: Destination peg identifier  
        auxiliary: Auxiliary peg identifier
        moves_list: List to store moves (optional)
    
    Returns:
        Number of moves required
    """
    if moves_list is None:
        moves_list = []
    
    if n == 1:
        move = f"Move disk 1 from {source} to {destination}"
        moves_list.append(move)
        print(move)
        return 1
    else:
        # Move n-1 disks from source to auxiliary
        moves1 = towers_of_hanoi(n-1, source, auxiliary, destination, moves_list)
        
        # Move the largest disk from source to destination
        move = f"Move disk {n} from {source} to {destination}"
        moves_list.append(move)
        print(move)
        
        # Move n-1 disks from auxiliary to destination
        moves2 = towers_of_hanoi(n-1, auxiliary, destination, source, moves_list)
        
        return moves1 + 1 + moves2

def visualize_hanoi_state(pegs, n):
    """
    Visualize the current state of the Towers of Hanoi
    
    Args:
        pegs: Dictionary containing the current state of each peg
        n: Total number of disks (for formatting)
    """
    print("\nCurrent state:")
    print("-" * (n * 4 + 10))
    
    # Print from top to bottom
    max_height = max(len(peg) for peg in pegs.values())
    
    for level in range(max_height - 1, -1, -1):
        line = ""
        for peg_name in ['A', 'B', 'C']:
            peg = pegs[peg_name]
            if level < len(peg):
                disk_size = peg[level]
                spaces = " " * (n - disk_size)
                disk_repr = "*" * (disk_size * 2)
                line += f"{spaces}{disk_repr}{spaces} | "
            else:
                spaces = " " * n
                line += f"{spaces}|{spaces} | "
        print(line)
    
    # Print base
    base_line = ""
    for peg_name in ['A', 'B', 'C']:
        base = "-" * (n * 2)
        base_line += f"{base} | "
    print(base_line)
    
    # Print peg labels
    label_line = ""
    for peg_name in ['A', 'B', 'C']:
        spaces = " " * (n - 1)
        label_line += f"{spaces}{peg_name}{spaces} | "
    print(label_line)
    print()

def towers_of_hanoi_with_visualization(n):
    """
    Solve Towers of Hanoi with step-by-step visualization
    
    Args:
        n: Number of disks
    """
    # Initialize pegs (A has all disks, B and C are empty)
    pegs = {
        'A': list(range(n, 0, -1)),  # Largest at bottom
        'B': [],
        'C': []
    }
    
    print(f"Solving Towers of Hanoi with {n} disks")
    print("Initial state:")
    visualize_hanoi_state(pegs, n)
    
    def move_disk(source, destination):
        """Move one disk from source to destination"""
        if pegs[source]:
            disk = pegs[source].pop()
            pegs[destination].append(disk)
            print(f"Move disk {disk} from {source} to {destination}")
            visualize_hanoi_state(pegs, n)
    
    def solve_recursive(n, source, destination, auxiliary):
        """Recursive solution with visualization"""
        if n == 1:
            move_disk(source, destination)
            return 1
        else:
            moves1 = solve_recursive(n-1, source, auxiliary, destination)
            move_disk(source, destination)
            moves2 = solve_recursive(n-1, auxiliary, destination, source)
            return moves1 + 1 + moves2
    
    total_moves = solve_recursive(n, 'A', 'C', 'B')
    print(f"\nPuzzle solved in {total_moves} moves!")
    print(f"Theoretical minimum moves: {2**n - 1}")
    
    return total_moves

def analyze_complexity(max_disks=10):
    """
    Analyze the time complexity by measuring moves for different disk counts
    
    Args:
        max_disks: Maximum number of disks to analyze
    """
    print("Towers of Hanoi Complexity Analysis")
    print("=" * 40)
    print(f"{'Disks':<6} {'Moves':<10} {'Formula':<12} {'Match':<8}")
    print("-" * 40)
    
    for n in range(1, max_disks + 1):
        moves_list = []
        actual_moves = towers_of_hanoi(n, 'A', 'C', 'B', moves_list)
        theoretical_moves = 2**n - 1
        match = "✓" if actual_moves == theoretical_moves else "✗"
        
        print(f"{n:<6} {actual_moves:<10} {theoretical_moves:<12} {match:<8}")

def interactive_demo():
    """
    Interactive demonstration of Towers of Hanoi
    """
    print("Welcome to Towers of Hanoi Solver!")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Solve without visualization\n"
                          "2. Solve with visualization\n"
                          "3. Complexity analysis\n"
                          "4. Exit\n"
                          "Enter your choice (1-4): ").strip()
            
            if choice == '1':
                n = int(input("Enter number of disks (1-10): "))
                if 1 <= n <= 10:
                    print(f"\nSolving {n} disks without visualization:")
                    moves = towers_of_hanoi(n, 'A', 'C', 'B')
                    print(f"Total moves: {moves}")
                else:
                    print("Please enter a number between 1 and 10")
                    
            elif choice == '2':
                n = int(input("Enter number of disks (1-5 recommended for visualization): "))
                if 1 <= n <= 8:
                    towers_of_hanoi_with_visualization(n)
                else:
                    print("Please enter a number between 1 and 8 for better visualization")
                    
            elif choice == '3':
                max_n = int(input("Enter maximum number of disks to analyze (1-15): "))
                if 1 <= max_n <= 15:
                    analyze_complexity(max_n)
                else:
                    print("Please enter a number between 1 and 15")
                    
            elif choice == '4':
                print("Thank you for using Towers of Hanoi Solver!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    # Quick demonstration
    print("Towers of Hanoi - Quick Demo")
    print("=" * 30)
    print("Solving for 3 disks:")
    towers_of_hanoi(3, 'A', 'C', 'B')
    print(f"Total moves: {2**3 - 1}")
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_demo() 