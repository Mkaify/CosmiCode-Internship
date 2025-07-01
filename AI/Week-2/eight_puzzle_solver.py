"""
Week 2: 8-Puzzle Problem Solver using DFS and BFS
Author: AI Learning Project
Date: 2024

This module implements the 8-puzzle problem solver using both DFS and BFS algorithms.
The 8-puzzle consists of a 3x3 grid with 8 numbered tiles and one empty space.
The goal is to arrange the tiles in numerical order.
"""

from collections import deque
import copy
import time

class PuzzleState:
    """Represents a state of the 8-puzzle"""
    
    def __init__(self, board, parent=None, move=None, depth=0):
        """
        Initialize puzzle state
        
        Args:
            board: 3x3 list representing the puzzle board (0 represents empty space)
            parent: Parent state (for path reconstruction)
            move: Move that led to this state
            depth: Depth from initial state
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.depth = depth
        self.empty_pos = self.find_empty_position()
    
    def find_empty_position(self):
        """Find the position of the empty space (0)"""
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return (i, j)
        return None
    
    def __eq__(self, other):
        """Check if two states are equal"""
        return self.board == other.board
    
    def __hash__(self):
        """Make state hashable for use in sets"""
        return hash(str(self.board))
    
    def __str__(self):
        """String representation of the puzzle"""
        result = ""
        for row in self.board:
            result += " ".join(str(x) if x != 0 else " " for x in row) + "\n"
        return result
    
    def is_goal(self, goal_state):
        """Check if current state is the goal state"""
        return self.board == goal_state.board
    
    def get_possible_moves(self):
        """Get all possible moves from current state"""
        moves = []
        row, col = self.empty_pos
        
        # Possible directions: up, down, left, right
        directions = [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]
        
        for dr, dc, direction in directions:
            new_row, new_col = row + dr, col + dc
            
            # Check if the move is within bounds
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                # Create new board state
                new_board = copy.deepcopy(self.board)
                # Swap empty space with the tile
                new_board[row][col], new_board[new_row][new_col] = \
                    new_board[new_row][new_col], new_board[row][col]
                
                moves.append((new_board, direction))
        
        return moves
    
    def get_successor_states(self):
        """Generate successor states"""
        successors = []
        possible_moves = self.get_possible_moves()
        
        for new_board, move in possible_moves:
            new_state = PuzzleState(new_board, self, move, self.depth + 1)
            successors.append(new_state)
        
        return successors
    
    def get_path(self):
        """Get the path from initial state to current state"""
        path = []
        current = self
        
        while current.parent is not None:
            path.append(current.move)
            current = current.parent
        
        path.reverse()
        return path

class EightPuzzleSolver:
    """Solver for the 8-puzzle problem"""
    
    def __init__(self, initial_state, goal_state):
        """
        Initialize the solver
        
        Args:
            initial_state: Initial puzzle configuration
            goal_state: Goal puzzle configuration
        """
        self.initial_state = PuzzleState(initial_state)
        self.goal_state = PuzzleState(goal_state)
        self.nodes_explored = 0
        self.max_depth = 0
    
    def is_solvable(self, board):
        """
        Check if the puzzle is solvable
        A puzzle is solvable if the number of inversions is even
        """
        flat_board = [tile for row in board for tile in row if tile != 0]
        inversions = 0
        
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        
        return inversions % 2 == 0
    
    def dfs_solve(self, max_depth=20):
        """
        Solve using Depth-First Search
        
        Args:
            max_depth: Maximum depth to search (to prevent infinite loops)
        
        Returns:
            tuple: (solution_path, statistics)
        """
        print("Solving 8-puzzle using DFS...")
        start_time = time.time()
        
        if not self.is_solvable(self.initial_state.board):
            return None, {"error": "Puzzle is not solvable"}
        
        stack = [self.initial_state]
        visited = set()
        self.nodes_explored = 0
        self.max_depth = 0
        
        while stack:
            current_state = stack.pop()
            self.nodes_explored += 1
            self.max_depth = max(self.max_depth, current_state.depth)
            
            # Check if we've reached the goal
            if current_state.is_goal(self.goal_state):
                end_time = time.time()
                solution_path = current_state.get_path()
                
                stats = {
                    "algorithm": "DFS",
                    "solution_found": True,
                    "path_length": len(solution_path),
                    "nodes_explored": self.nodes_explored,
                    "max_depth_reached": self.max_depth,
                    "time_taken": end_time - start_time,
                    "solution_depth": current_state.depth
                }
                
                return solution_path, stats
            
            # Skip if already visited or too deep
            if current_state in visited or current_state.depth >= max_depth:
                continue
            
            visited.add(current_state)
            
            # Add successor states to stack
            successors = current_state.get_successor_states()
            # Reverse to maintain left-to-right exploration order
            for successor in reversed(successors):
                if successor not in visited:
                    stack.append(successor)
        
        end_time = time.time()
        stats = {
            "algorithm": "DFS",
            "solution_found": False,
            "nodes_explored": self.nodes_explored,
            "max_depth_reached": self.max_depth,
            "time_taken": end_time - start_time,
            "error": f"No solution found within depth {max_depth}"
        }
        
        return None, stats
    
    def bfs_solve(self):
        """
        Solve using Breadth-First Search
        
        Returns:
            tuple: (solution_path, statistics)
        """
        print("Solving 8-puzzle using BFS...")
        start_time = time.time()
        
        if not self.is_solvable(self.initial_state.board):
            return None, {"error": "Puzzle is not solvable"}
        
        queue = deque([self.initial_state])
        visited = set()
        self.nodes_explored = 0
        self.max_depth = 0
        
        while queue:
            current_state = queue.popleft()
            self.nodes_explored += 1
            self.max_depth = max(self.max_depth, current_state.depth)
            
            # Check if we've reached the goal
            if current_state.is_goal(self.goal_state):
                end_time = time.time()
                solution_path = current_state.get_path()
                
                stats = {
                    "algorithm": "BFS",
                    "solution_found": True,
                    "path_length": len(solution_path),
                    "nodes_explored": self.nodes_explored,
                    "max_depth_reached": self.max_depth,
                    "time_taken": end_time - start_time,
                    "solution_depth": current_state.depth
                }
                
                return solution_path, stats
            
            # Skip if already visited
            if current_state in visited:
                continue
            
            visited.add(current_state)
            
            # Add successor states to queue
            successors = current_state.get_successor_states()
            for successor in successors:
                if successor not in visited:
                    queue.append(successor)
        
        end_time = time.time()
        stats = {
            "algorithm": "BFS",
            "solution_found": False,
            "nodes_explored": self.nodes_explored,
            "max_depth_reached": self.max_depth,
            "time_taken": end_time - start_time,
            "error": "No solution found"
        }
        
        return None, stats
    
    def print_solution(self, path, stats):
        """Print the solution path and statistics"""
        if path is None:
            print(f"No solution found using {stats['algorithm']}")
            print(f"Error: {stats.get('error', 'Unknown error')}")
            return
        
        print(f"\nSolution found using {stats['algorithm']}!")
        print(f"Solution path ({len(path)} moves): {' -> '.join(path)}")
        print(f"Solution depth: {stats['solution_depth']}")
        print(f"Nodes explored: {stats['nodes_explored']}")
        print(f"Max depth reached: {stats['max_depth_reached']}")
        print(f"Time taken: {stats['time_taken']:.4f} seconds")

def demonstrate_eight_puzzle():
    """Demonstrate the 8-puzzle solver"""
    print("=== 8-Puzzle Solver Demonstration ===\n")
    
    # Define initial and goal states
    # Initial state (solvable puzzle)
    initial = [
        [1, 2, 3],
        [4, 0, 6],
        [7, 5, 8]
    ]
    
    # Goal state
    goal = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    print("Initial State:")
    initial_state = PuzzleState(initial)
    print(initial_state)
    
    print("Goal State:")
    goal_state = PuzzleState(goal)
    print(goal_state)
    
    # Create solver
    solver = EightPuzzleSolver(initial, goal)
    
    # Solve using BFS
    print("=" * 50)
    bfs_path, bfs_stats = solver.bfs_solve()
    solver.print_solution(bfs_path, bfs_stats)
    
    # Solve using DFS
    print("\n" + "=" * 50)
    dfs_path, dfs_stats = solver.dfs_solve(max_depth=15)
    solver.print_solution(dfs_path, dfs_stats)
    
    # Compare algorithms
    print("\n" + "=" * 50)
    print("Algorithm Comparison:")
    if bfs_stats['solution_found'] and dfs_stats['solution_found']:
        print(f"BFS found solution in {bfs_stats['solution_depth']} moves, explored {bfs_stats['nodes_explored']} nodes")
        print(f"DFS found solution in {dfs_stats['solution_depth']} moves, explored {dfs_stats['nodes_explored']} nodes")
        print(f"BFS time: {bfs_stats['time_taken']:.4f}s, DFS time: {dfs_stats['time_taken']:.4f}s")

def test_different_puzzles():
    """Test solver with different puzzle configurations"""
    print("\n=== Testing Different Puzzle Configurations ===\n")
    
    test_cases = [
        {
            "name": "Easy Puzzle (2 moves)",
            "initial": [[1, 2, 3], [4, 5, 6], [7, 0, 8]],
            "goal": [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        },
        {
            "name": "Medium Puzzle",
            "initial": [[1, 2, 3], [0, 4, 6], [7, 5, 8]],
            "goal": [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        },
        {
            "name": "Unsolvable Puzzle",
            "initial": [[1, 2, 3], [4, 5, 6], [8, 7, 0]],
            "goal": [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        solver = EightPuzzleSolver(test_case['initial'], test_case['goal'])
        
        # Try BFS first (usually faster for short solutions)
        path, stats = solver.bfs_solve()
        
        if stats.get('solution_found', False):
            print(f"  Solution: {len(path)} moves, {stats['nodes_explored']} nodes explored")
        else:
            print(f"  {stats.get('error', 'No solution found')}")
        
        print()

if __name__ == "__main__":
    print("Week 2: 8-Puzzle Problem Solver\n")
    
    demonstrate_eight_puzzle()
    test_different_puzzles()
    
    print("=== Key Insights ===")
    print("1. BFS guarantees optimal solution (shortest path)")
    print("2. DFS may find solution faster but not guaranteed to be optimal")
    print("3. Solvability check prevents infinite search")
    print("4. State space can be very large (9!/2 ≈ 181,440 possible states)")
    print("5. Memory usage differs: BFS uses more memory, DFS uses less") 