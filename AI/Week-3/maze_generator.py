"""
Random Maze Generator and Display

This program generates random mazes using different algorithms and provides
various visualization options. The mazes can be displayed in the console
or optionally saved to files.

Algorithms implemented:
1. Recursive Backtracking (DFS-based)
2. Randomized Kruskal's Algorithm
3. Simple Random Walk

Features:
- Multiple maze generation algorithms
- ASCII art visualization
- Configurable maze dimensions
- Start and end point marking
- Maze solving capability
"""

import random
import time
from collections import deque

class Maze:
    def __init__(self, width, height):
        """
        Initialize maze with given dimensions
        
        Args:
            width: Width of the maze (odd number recommended)
            height: Height of the maze (odd number recommended)
        """
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.grid = [['#' for _ in range(self.width)] for _ in range(self.height)]
        self.start = (1, 1)
        self.end = (self.height - 2, self.width - 2)
        
    def display(self, path=None, show_solution=False):
        """
        Display the maze in ASCII format
        
        Args:
            path: Optional path to highlight
            show_solution: Whether to show the solution path
        """
        print(f"\nMaze ({self.height} x {self.width}):")
        print("Legend: # = Wall, . = Path, S = Start, E = End, * = Solution")
        print("-" * (self.width + 2))
        
        for i, row in enumerate(self.grid):
            print("|", end="")
            for j, cell in enumerate(row):
                if (i, j) == self.start:
                    print("S", end="")
                elif (i, j) == self.end:
                    print("E", end="")
                elif show_solution and path and (i, j) in path:
                    print("*", end="")
                else:
                    print(cell, end="")
            print("|")
        print("-" * (self.width + 2))
    
    def is_valid_cell(self, row, col):
        """Check if a cell is within maze bounds"""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def get_neighbors(self, row, col, distance=2):
        """
        Get valid neighboring cells at a given distance
        
        Args:
            row, col: Current position
            distance: Distance to neighbors (2 for wall-separated cells)
        """
        neighbors = []
        directions = [(0, distance), (distance, 0), (0, -distance), (-distance, 0)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_cell(new_row, new_col):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def carve_path(self, from_cell, to_cell):
        """
        Carve a path between two cells by removing the wall between them
        
        Args:
            from_cell: Starting cell (row, col)
            to_cell: Ending cell (row, col)
        """
        row1, col1 = from_cell
        row2, col2 = to_cell
        
        # Mark both cells as paths
        self.grid[row1][col1] = '.'
        self.grid[row2][col2] = '.'
        
        # Mark the wall between them as a path
        wall_row = (row1 + row2) // 2
        wall_col = (col1 + col2) // 2
        self.grid[wall_row][wall_col] = '.'

class MazeGenerator:
    def __init__(self, maze):
        """
        Initialize maze generator
        
        Args:
            maze: Maze object to generate
        """
        self.maze = maze
    
    def recursive_backtracking(self, animate=False):
        """
        Generate maze using recursive backtracking algorithm
        
        Args:
            animate: Whether to show animation during generation
        """
        # Start from the top-left corner
        stack = [self.maze.start]
        visited = {self.maze.start}
        self.maze.grid[self.maze.start[0]][self.maze.start[1]] = '.'
        
        while stack:
            current = stack[-1]
            row, col = current
            
            # Get unvisited neighbors
            neighbors = self.maze.get_neighbors(row, col)
            unvisited_neighbors = [n for n in neighbors if n not in visited]
            
            if unvisited_neighbors:
                # Choose a random unvisited neighbor
                next_cell = random.choice(unvisited_neighbors)
                
                # Carve path to the neighbor
                self.maze.carve_path(current, next_cell)
                
                # Mark as visited and add to stack
                visited.add(next_cell)
                stack.append(next_cell)
                
                if animate:
                    self.maze.display()
                    time.sleep(0.1)
            else:
                # Backtrack
                stack.pop()
    
    def randomized_kruskals(self):
        """
        Generate maze using randomized Kruskal's algorithm
        """
        # Initialize all cells as walls
        cells = []
        for i in range(1, self.maze.height, 2):
            for j in range(1, self.maze.width, 2):
                cells.append((i, j))
                self.maze.grid[i][j] = '.'
        
        # Create all possible edges between adjacent cells
        edges = []
        for row, col in cells:
            neighbors = self.maze.get_neighbors(row, col)
            for neighbor in neighbors:
                if neighbor in cells:
                    edges.append(((row, col), neighbor))
        
        # Randomize edges
        random.shuffle(edges)
        
        # Union-Find data structure
        parent = {cell: cell for cell in cells}
        
        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]
        
        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                parent[root1] = root2
                return True
            return False
        
        # Process edges
        for edge in edges:
            cell1, cell2 = edge
            if union(cell1, cell2):
                # Carve path between cells
                self.maze.carve_path(cell1, cell2)
    
    def simple_random_walk(self):
        """
        Generate maze using simple random walk
        """
        current = self.maze.start
        visited = {current}
        self.maze.grid[current[0]][current[1]] = '.'
        
        # Perform random walk
        for _ in range(self.maze.width * self.maze.height // 4):
            neighbors = self.maze.get_neighbors(current[0], current[1])
            
            if neighbors:
                next_cell = random.choice(neighbors)
                self.maze.carve_path(current, next_cell)
                visited.add(next_cell)
                current = next_cell

class MazeSolver:
    def __init__(self, maze):
        """
        Initialize maze solver
        
        Args:
            maze: Maze object to solve
        """
        self.maze = maze
    
    def bfs_solve(self):
        """
        Solve maze using Breadth-First Search
        
        Returns:
            list: Path from start to end, or None if no solution
        """
        queue = deque([(self.maze.start, [self.maze.start])])
        visited = {self.maze.start}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            (row, col), path = queue.popleft()
            
            if (row, col) == self.maze.end:
                return path
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (self.maze.is_valid_cell(new_row, new_col) and
                    (new_row, new_col) not in visited and
                    self.maze.grid[new_row][new_col] in ['.', 'S', 'E']):
                    
                    visited.add((new_row, new_col))
                    queue.append(((new_row, new_col), path + [(new_row, new_col)]))
        
        return None

def generate_and_display_maze(width, height, algorithm='backtracking', solve=False, animate=False):
    """
    Generate and display a maze
    
    Args:
        width, height: Maze dimensions
        algorithm: Generation algorithm ('backtracking', 'kruskal', 'random_walk')
        solve: Whether to solve the maze after generation
        animate: Whether to animate the generation process
    """
    print(f"Generating {height}x{width} maze using {algorithm} algorithm...")
    
    # Create maze
    maze = Maze(width, height)
    generator = MazeGenerator(maze)
    
    # Generate maze based on selected algorithm
    start_time = time.time()
    
    if algorithm == 'backtracking':
        generator.recursive_backtracking(animate)
    elif algorithm == 'kruskal':
        generator.randomized_kruskals()
    elif algorithm == 'random_walk':
        generator.simple_random_walk()
    else:
        print(f"Unknown algorithm: {algorithm}")
        return
    
    generation_time = time.time() - start_time
    
    # Display the generated maze
    maze.display()
    print(f"Generation time: {generation_time:.3f} seconds")
    
    # Solve if requested
    if solve:
        print("\nSolving maze...")
        solver = MazeSolver(maze)
        solution_path = solver.bfs_solve()
        
        if solution_path:
            print(f"Solution found! Path length: {len(solution_path)}")
            maze.display(path=solution_path, show_solution=True)
        else:
            print("No solution found!")
    
    return maze

def compare_algorithms():
    """
    Compare different maze generation algorithms
    """
    print("Comparing Maze Generation Algorithms")
    print("=" * 50)
    
    width, height = 21, 21
    algorithms = ['backtracking', 'kruskal', 'random_walk']
    
    for algorithm in algorithms:
        print(f"\n{algorithm.upper()} ALGORITHM:")
        print("-" * 30)
        
        maze = Maze(width, height)
        generator = MazeGenerator(maze)
        
        start_time = time.time()
        
        if algorithm == 'backtracking':
            generator.recursive_backtracking()
        elif algorithm == 'kruskal':
            generator.randomized_kruskals()
        elif algorithm == 'random_walk':
            generator.simple_random_walk()
        
        generation_time = time.time() - start_time
        
        # Quick display (smaller version)
        print(f"Generation time: {generation_time:.3f} seconds")
        
        # Count paths vs walls
        paths = sum(row.count('.') for row in maze.grid)
        walls = sum(row.count('#') for row in maze.grid)
        
        print(f"Paths: {paths}, Walls: {walls}")
        print(f"Path percentage: {paths / (paths + walls) * 100:.1f}%")

def interactive_demo():
    """
    Interactive demonstration of maze generation
    """
    print("Welcome to Random Maze Generator!")
    print("=" * 40)
    
    while True:
        try:
            choice = input("\nChoose an option:\n"
                          "1. Generate maze with recursive backtracking\n"
                          "2. Generate maze with Kruskal's algorithm\n"
                          "3. Generate maze with random walk\n"
                          "4. Compare all algorithms\n"
                          "5. Custom maze generation\n"
                          "6. Exit\n"
                          "Enter your choice (1-6): ").strip()
            
            if choice in ['1', '2', '3']:
                algorithms = ['backtracking', 'kruskal', 'random_walk']
                algorithm = algorithms[int(choice) - 1]
                
                width = int(input("Enter maze width (odd number, 5-50): ") or "21")
                height = int(input("Enter maze height (odd number, 5-50): ") or "21")
                
                solve_maze = input("Solve the maze after generation? (y/n): ").lower().strip() == 'y'
                animate = False
                
                if algorithm == 'backtracking':
                    animate = input("Show generation animation? (y/n): ").lower().strip() == 'y'
                
                generate_and_display_maze(width, height, algorithm, solve_maze, animate)
                
            elif choice == '4':
                compare_algorithms()
                
            elif choice == '5':
                print("Custom maze generation:")
                width = int(input("Enter maze width: ") or "21")
                height = int(input("Enter maze height: ") or "21")
                
                print("Available algorithms: backtracking, kruskal, random_walk")
                algorithm = input("Enter algorithm: ").strip() or "backtracking"
                
                solve_maze = input("Solve the maze? (y/n): ").lower().strip() == 'y'
                
                generate_and_display_maze(width, height, algorithm, solve_maze)
                
            elif choice == '6':
                print("Thank you for using Maze Generator!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    # Quick demonstration
    print("Random Maze Generator - Quick Demo")
    print("=" * 35)
    
    # Generate a small maze
    demo_maze = generate_and_display_maze(11, 11, 'backtracking', solve=True)
    
    print("\n" + "=" * 50)
    
    # Run interactive demo
    interactive_demo() 