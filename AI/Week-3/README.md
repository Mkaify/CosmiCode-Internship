# Week 3: Problem Solving & AI Concepts (Intermediate)

This folder contains implementations of classic AI and computer science problems demonstrating various problem-solving strategies including recursion, backtracking, and game development.

## 📁 Files Overview

| File | Description |
|------|-------------|
| `towers_of_hanoi.py` | Recursive solution to the classic Towers of Hanoi puzzle |
| `tic_tac_toe.py` | Player vs Player Tic-Tac-Toe game with tournament mode |
| `n_queens.py` | N-Queens problem solver using backtracking algorithm |
| `maze_generator.py` | Random maze generation with multiple algorithms and solving |
| `intelligent_agents_research.md` | Research note on Intelligent Agents and their types |
| `requirements.txt` | Python package dependencies |
| `README.md` | This documentation file |

## 🎯 Learning Objectives

By the end of this week, you will understand:

1. **Recursive Problem Solving**: Using recursion to solve complex problems
2. **Backtracking Algorithms**: Systematic approach to finding solutions
3. **Game Development**: Creating interactive games with proper logic
4. **Algorithm Analysis**: Understanding time and space complexity
5. **AI Concepts**: Fundamentals of intelligent agents

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Basic understanding of recursion and algorithms

### Installation

1. Navigate to the Week-3 directory
2. (Optional) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Problem Implementations

### 1. Towers of Hanoi (`towers_of_hanoi.py`)

**Problem Description:**
Classic puzzle with three pegs and n disks. Move all disks from source to destination following the rules.

**Key Features:**
- Recursive solution implementation
- Step-by-step visualization
- Complexity analysis
- Interactive demonstrations

**Run the program:**
```bash
python towers_of_hanoi.py
```

**Algorithm Details:**
- Time Complexity: O(2^n)
- Space Complexity: O(n)
- Minimum moves: 2^n - 1

### 2. Tic-Tac-Toe Game (`tic_tac_toe.py`)

**Problem Description:**
Classic 3x3 grid game where players take turns trying to get three marks in a row.

**Key Features:**
- Player vs Player gameplay
- Input validation and error handling
- Score tracking across multiple rounds
- Tournament mode

**Run the program:**
```bash
python tic_tac_toe.py
```

**Game Features:**
- Clean console interface
- Win/draw detection
- Multiple game rounds
- Final score summary

### 3. N-Queens Problem (`n_queens.py`)

**Problem Description:**
Place N chess queens on an N×N chessboard so that no two queens attack each other.

**Key Features:**
- Backtracking algorithm implementation
- Find single or all solutions
- Step-by-step visualization
- Complexity analysis

**Run the program:**
```bash
python n_queens.py
```

**Algorithm Details:**
- Time Complexity: O(N!)
- Space Complexity: O(N)
- Classic constraint satisfaction problem

### 4. Maze Generator (`maze_generator.py`)

**Problem Description:**
Generate random mazes using different algorithms and provide solving capabilities.

**Key Features:**
- Multiple generation algorithms
- ASCII art visualization
- Maze solving with BFS
- Algorithm comparison

**Run the program:**
```bash
python maze_generator.py
```

**Algorithms Implemented:**
1. Recursive Backtracking (DFS-based)
2. Randomized Kruskal's Algorithm
3. Simple Random Walk

### 5. Research Note (`intelligent_agents_research.md`)

**Content:**
Comprehensive overview of intelligent agents and their classification in AI systems.

**Topics Covered:**
- Definition of intelligent agents
- Types of agents (Reflex, Model-based, Goal-based, Utility-based, Learning)
- Real-world applications and examples

## 🎮 Interactive Examples

### Example 1: Towers of Hanoi
```python
from towers_of_hanoi import towers_of_hanoi

# Solve for 3 disks
moves = towers_of_hanoi(3, 'A', 'C', 'B')
print(f"Solved in {moves} moves")
```

### Example 2: N-Queens
```python
from n_queens import NQueens

# Solve 8-Queens problem
queens = NQueens(8)
if queens.solve():
    queens.print_board()
```

### Example 3: Maze Generation
```python
from maze_generator import generate_and_display_maze

# Generate and solve a 15x15 maze
maze = generate_and_display_maze(15, 15, 'backtracking', solve=True)
```

## 📚 Algorithm Analysis

### Complexity Comparison

| Algorithm | Time Complexity | Space Complexity | Problem Type |
|-----------|----------------|------------------|--------------|
| Towers of Hanoi | O(2^n) | O(n) | Recursive |
| N-Queens | O(N!) | O(N) | Backtracking |
| Maze Generation | O(V + E) | O(V) | Graph-based |
| Tic-Tac-Toe | O(1) per move | O(1) | Game Logic |

### Problem-Solving Strategies

**Recursion (Towers of Hanoi):**
- Break problem into smaller subproblems
- Base case and recursive case
- Natural for problems with self-similar structure

**Backtracking (N-Queens):**
- Systematic exploration of solution space
- Prune invalid paths early
- Ideal for constraint satisfaction problems

**Game Theory (Tic-Tac-Toe):**
- State management
- Turn-based logic
- Win condition checking

## 🧪 Exercises and Challenges

### Basic Exercises
1. Modify Towers of Hanoi to work with 4 pegs
2. Add AI player to Tic-Tac-Toe using minimax
3. Implement different heuristics for N-Queens
4. Create maze solver using A* algorithm

### Advanced Challenges
1. Visualize Towers of Hanoi using graphics
2. Create multiplayer Tic-Tac-Toe over network
3. Solve N-Queens for larger boards (N > 12)
4. Generate 3D mazes

### Extension Projects
1. Compare recursive vs iterative solutions
2. Implement parallel backtracking
3. Create web-based versions of the games
4. Add machine learning to improve algorithms

## ⚡ Performance Tips

### Optimization Strategies
1. **Memoization**: Cache results for repeated subproblems
2. **Pruning**: Eliminate invalid paths early in backtracking
3. **Heuristics**: Use domain knowledge to guide search
4. **Iterative Deepening**: Combine benefits of DFS and BFS

### Memory Management
1. Use generators for large search spaces
2. Implement tail recursion where possible
3. Clear data structures when no longer needed
4. Consider iterative versions for deep recursions

## 🐛 Common Pitfalls

1. **Stack Overflow**: Deep recursions in Towers of Hanoi
2. **Infinite Loops**: Improper backtracking implementation
3. **State Management**: Not properly resetting game states
4. **Input Validation**: Not handling invalid user inputs

## 📈 Next Steps

After mastering these concepts, explore:
- Advanced search algorithms (A*, beam search)
- Machine learning applications
- Parallel and distributed algorithms
- Real-time and anytime algorithms
- Constraint programming

## 🤝 Contributing

Feel free to:
- Add more visualization options
- Implement additional algorithms
- Optimize existing solutions
- Create new problem variations

## 📄 License

This educational content is provided for learning purposes. Feel free to use and modify for educational goals.

---

**Happy Problem Solving! 🧩**

*Remember: These classic problems form the foundation of computer science and AI. Understanding their solutions and trade-offs will help you tackle more complex problems in the future.* 