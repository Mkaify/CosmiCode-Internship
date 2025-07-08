"""
N-Queens Problem Solver using Backtracking

The N-Queens problem is the challenge of placing N chess queens on an N×N chessboard
so that no two queens attack each other. This means no two queens can be in the same
row, column, or diagonal.

Algorithm: Backtracking
Time Complexity: O(N!)
Space Complexity: O(N)
"""

class NQueens:
    def __init__(self, n):
        """
        Initialize the N-Queens solver
        
        Args:
            n: Size of the chessboard (n x n)
        """
        self.n = n
        self.board = [[0 for _ in range(n)] for _ in range(n)]
        self.solutions = []
        
    def print_board(self, board=None):
        """
        Print the current board state
        
        Args:
            board: Optional board to print (uses self.board if None)
        """
        if board is None:
            board = self.board
            
        print("\n" + "+" + "-" * (self.n * 4 - 1) + "+")
        for row in board:
            print("|", end="")
            for cell in row:
                if cell == 1:
                    print(" Q ", end="|")
                else:
                    print("   ", end="|")
            print("\n" + "+" + "-" * (self.n * 4 - 1) + "+")
        print()
    
    def is_safe(self, row, col):
        """
        Check if it's safe to place a queen at board[row][col]
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            bool: True if safe, False otherwise
        """
        # Check this column on upper rows
        for i in range(row):
            if self.board[i][col] == 1:
                return False
        
        # Check upper diagonal on left side
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if self.board[i][j] == 1:
                return False
        
        # Check upper diagonal on right side
        for i, j in zip(range(row-1, -1, -1), range(col+1, self.n)):
            if self.board[i][j] == 1:
                return False
        
        return True
    
    def solve_nqueens_util(self, row, show_steps=False):
        """
        Utility function to solve N-Queens using backtracking
        
        Args:
            row: Current row being processed
            show_steps: Whether to show intermediate steps
            
        Returns:
            bool: True if solution exists, False otherwise
        """
        # Base case: all queens are placed
        if row >= self.n:
            # Store the solution
            solution = [row[:] for row in self.board]
            self.solutions.append(solution)
            return True
        
        # Try placing queen in each column of current row
        for col in range(self.n):
            if self.is_safe(row, col):
                # Place queen
                self.board[row][col] = 1
                
                if show_steps:
                    print(f"Placing queen at position ({row}, {col})")
                    self.print_board()
                    input("Press Enter to continue...")
                
                # Recursively place queens in remaining rows
                if self.solve_nqueens_util(row + 1, show_steps):
                    return True
                
                # If placing queen at [row][col] doesn't lead to solution,
                # remove the queen (BACKTRACK)
                self.board[row][col] = 0
                
                if show_steps:
                    print(f"Backtracking from position ({row}, {col})")
                    self.print_board()
                    input("Press Enter to continue...")
        
        # If no column works for this row, return False
        return False
    
    def solve_all_solutions(self, row=0):
        """
        Find all possible solutions to the N-Queens problem
        
        Args:
            row: Current row being processed
        """
        # Base case: all queens are placed
        if row >= self.n:
            # Store the solution
            solution = [row[:] for row in self.board]
            self.solutions.append(solution)
            return
        
        # Try placing queen in each column of current row
        for col in range(self.n):
            if self.is_safe(row, col):
                # Place queen
                self.board[row][col] = 1
                
                # Recursively place queens in remaining rows
                self.solve_all_solutions(row + 1)
                
                # Backtrack
                self.board[row][col] = 0
    
    def solve(self, find_all=False, show_steps=False):
        """
        Solve the N-Queens problem
        
        Args:
            find_all: Whether to find all solutions or just one
            show_steps: Whether to show solving steps
            
        Returns:
            bool: True if at least one solution exists
        """
        self.solutions = []
        self.board = [[0 for _ in range(self.n)] for _ in range(self.n)]
        
        if find_all:
            self.solve_all_solutions()
            return len(self.solutions) > 0
        else:
            return self.solve_nqueens_util(0, show_steps)
    
    def print_solutions(self, max_solutions=None):
        """
        Print all found solutions
        
        Args:
            max_solutions: Maximum number of solutions to print
        """
        if not self.solutions:
            print("No solutions found!")
            return
        
        solutions_to_show = self.solutions[:max_solutions] if max_solutions else self.solutions
        
        print(f"\nFound {len(self.solutions)} solution(s) for {self.n}-Queens problem")
        print(f"Showing {len(solutions_to_show)} solution(s):\n")
        
        for i, solution in enumerate(solutions_to_show, 1):
            print(f"Solution {i}:")
            self.print_board(solution)
    
    def get_solution_statistics(self):
        """
        Get statistics about the solutions
        
        Returns:
            dict: Statistics about the solutions
        """
        if not self.solutions:
            return {"total_solutions": 0}
        
        stats = {
            "total_solutions": len(self.solutions),
            "board_size": self.n,
            "unique_positions": set()
        }
        
        # Collect unique queen positions
        for solution in self.solutions:
            positions = []
            for row in range(self.n):
                for col in range(self.n):
                    if solution[row][col] == 1:
                        positions.append((row, col))
            stats["unique_positions"].add(tuple(positions))
        
        return stats

def analyze_nqueens_complexity():
    """
    Analyze the N-Queens problem for different board sizes
    """
    print("N-Queens Problem Analysis")
    print("=" * 40)
    print(f"{'N':<4} {'Solutions':<12} {'Time (approx)':<15}")
    print("-" * 40)
    
    known_solutions = {
        1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92,
        9: 352, 10: 724, 11: 2680, 12: 14200
    }
    
    for n in range(1, 9):  # Up to 8 for reasonable computation time
        if n in known_solutions:
            print(f"{n:<4} {known_solutions[n]:<12} {'<1s' if n <= 8 else '>1s':<15}")
        else:
            queens = NQueens(n)
            queens.solve(find_all=True)
            solutions_count = len(queens.solutions)
            print(f"{n:<4} {solutions_count:<12} {'<1s' if n <= 8 else '>1s':<15}")

def interactive_demo():
    """
    Interactive demonstration of N-Queens solver
    """
    print("Welcome to N-Queens Solver!")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Solve for specific N (show one solution)\n"
                          "2. Solve with step-by-step visualization\n"
                          "3. Find all solutions\n"
                          "4. Complexity analysis\n"
                          "5. Exit\n"
                          "Enter your choice (1-5): ").strip()
            
            if choice == '1':
                n = int(input("Enter the size of chessboard (4-12 recommended): "))
                if n <= 0:
                    print("Please enter a positive number.")
                    continue
                
                queens = NQueens(n)
                print(f"\nSolving {n}-Queens problem...")
                
                if queens.solve():
                    print(f"Solution found for {n}-Queens:")
                    queens.print_board()
                else:
                    print(f"No solution exists for {n}-Queens problem.")
                    
            elif choice == '2':
                n = int(input("Enter the size of chessboard (4-6 recommended for visualization): "))
                if n <= 0:
                    print("Please enter a positive number.")
                    continue
                
                queens = NQueens(n)
                print(f"\nSolving {n}-Queens with step-by-step visualization...")
                print("Press Enter after each step to continue.")
                
                if queens.solve(show_steps=True):
                    print(f"Solution found for {n}-Queens!")
                else:
                    print(f"No solution exists for {n}-Queens problem.")
                    
            elif choice == '3':
                n = int(input("Enter the size of chessboard (4-8 recommended): "))
                if n <= 0:
                    print("Please enter a positive number.")
                    continue
                
                queens = NQueens(n)
                print(f"\nFinding all solutions for {n}-Queens problem...")
                
                queens.solve(find_all=True)
                
                if queens.solutions:
                    stats = queens.get_solution_statistics()
                    print(f"Found {stats['total_solutions']} solutions!")
                    
                    show_all = input("Show all solutions? (y/n): ").lower().strip()
                    if show_all in ['y', 'yes']:
                        max_show = None
                        if stats['total_solutions'] > 10:
                            max_show = int(input(f"Show how many solutions? (max {stats['total_solutions']}): "))
                        queens.print_solutions(max_show)
                    else:
                        queens.print_solutions(1)
                else:
                    print(f"No solutions exist for {n}-Queens problem.")
                    
            elif choice == '4':
                print("\nAnalyzing N-Queens complexity...")
                analyze_nqueens_complexity()
                
            elif choice == '5':
                print("Thank you for using N-Queens Solver!")
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
    print("N-Queens Problem - Quick Demo")
    print("=" * 30)
    
    # Solve 4-Queens as example
    queens = NQueens(4)
    print("Solving 4-Queens problem:")
    
    if queens.solve():
        print("Solution found:")
        queens.print_board()
    else:
        print("No solution found.")
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_demo() 