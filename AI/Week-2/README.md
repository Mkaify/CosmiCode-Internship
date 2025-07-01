# Week 2: Search Algorithms in AI

This folder contains implementations of various search algorithms as part of the AI learning curriculum. The implementations range from basic linear and binary search to advanced graph traversal algorithms and the classic 8-puzzle problem.

## 📁 Files Overview

| File | Description |
|------|-------------|
| `linear_binary_search.py` | Linear and Binary Search implementations with performance comparison |
| `graph_search_algorithms.py` | DFS and BFS implementations using adjacency list representation |
| `eight_puzzle_solver.py` | 8-puzzle problem solver using DFS and BFS |
| `search_algorithms_report.md` | Comprehensive report on A* and Greedy Best-First Search |
| `requirements.txt` | Python package dependencies |
| `README.md` | This documentation file |

## 🎯 Learning Objectives

By the end of this week, you will understand:

1. **Basic Search Algorithms**: Linear vs Binary search trade-offs
2. **Graph Traversal**: DFS and BFS algorithms and their applications
3. **Problem Solving**: Using search algorithms to solve real problems (8-puzzle)
4. **Advanced Concepts**: A* and Greedy Best-First Search theory and applications

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Basic understanding of data structures (lists, graphs)

### Installation

1. Clone or download the repository
2. Navigate to the Week-2 directory
3. (Optional) Install additional packages for enhanced features:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Algorithm Implementations

### 1. Linear and Binary Search (`linear_binary_search.py`)

**Algorithms Implemented:**
- Linear Search (O(n))
- Binary Search - Iterative (O(log n))
- Binary Search - Recursive (O(log n))

**Run the demonstration:**
```bash
python linear_binary_search.py
```

**Key Features:**
- Performance comparison between algorithms
- Examples with both numbers and strings
- Time complexity analysis

### 2. Graph Search Algorithms (`graph_search_algorithms.py`)

**Algorithms Implemented:**
- Depth-First Search (DFS) - Iterative and Recursive
- Breadth-First Search (BFS)
- Shortest Path finding using BFS

**Run the demonstration:**
```bash
python graph_search_algorithms.py
```

**Key Features:**
- Adjacency list graph representation
- Both directed and undirected graph examples
- Target search functionality
- Path reconstruction

### 3. 8-Puzzle Solver (`eight_puzzle_solver.py`)

**Problem Description:**
The 8-puzzle consists of a 3×3 grid with 8 numbered tiles and one empty space. The goal is to arrange tiles in numerical order.

**Algorithms Used:**
- DFS with depth limiting
- BFS for optimal solutions
- Solvability checking

**Run the demonstration:**
```bash
python eight_puzzle_solver.py
```

**Key Features:**
- Multiple test cases
- Performance statistics
- Path reconstruction
- Algorithm comparison

## 📖 Theoretical Understanding

### Search Algorithm Comparison

| Algorithm | Time Complexity | Space Complexity | Optimal | Complete |
|-----------|----------------|------------------|---------|----------|
| Linear Search | O(n) | O(1) | Yes* | Yes |
| Binary Search | O(log n) | O(1) | Yes* | Yes |
| DFS | O(V + E) | O(V) | No | No** |
| BFS | O(V + E) | O(V) | Yes*** | Yes |

*For their respective problem domains  
**In infinite spaces  
***For unweighted graphs

### When to Use Each Algorithm

**Linear Search:**
- Small datasets
- Unsorted data
- Simple implementation needed

**Binary Search:**
- Large sorted datasets
- Frequent search operations
- Memory efficiency important

**DFS:**
- Memory is limited
- Deep solutions expected
- Backtracking problems

**BFS:**
- Shortest path needed
- Shallow solutions expected
- Complete exploration required

## 🎮 Interactive Examples

### Example 1: Search Performance
```python
from linear_binary_search import linear_search, binary_search

# Test data
numbers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
target = 13

# Linear search
linear_result = linear_search(numbers, target)
print(f"Linear search found {target} at index {linear_result}")

# Binary search
binary_result = binary_search(numbers, target)
print(f"Binary search found {target} at index {binary_result}")
```

### Example 2: Graph Traversal
```python
from graph_search_algorithms import Graph

# Create a simple graph
g = Graph()
g.add_edge('A', 'B')
g.add_edge('A', 'C')
g.add_edge('B', 'D')

# Perform DFS and BFS
dfs_path = g.dfs('A')
bfs_path = g.bfs('A')

print(f"DFS path: {dfs_path}")
print(f"BFS path: {bfs_path}")
```

### Example 3: 8-Puzzle Solving
```python
from eight_puzzle_solver import EightPuzzleSolver

# Define puzzle states
initial = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

# Solve using BFS
solver = EightPuzzleSolver(initial, goal)
path, stats = solver.bfs_solve()

if path:
    print(f"Solution found in {len(path)} moves!")
    print(f"Moves: {path}")
else:
    print("No solution found")
```

## 📚 Additional Resources

### Study Materials
1. **search_algorithms_report.md** - Detailed analysis of A* and Greedy Best-First Search
2. Algorithm visualizations available online
3. Practice problems for each algorithm type

### Recommended Reading
- "Artificial Intelligence: A Modern Approach" by Russell & Norvig
- "Introduction to Algorithms" by Cormen, Leiserson, Rivest & Stein
- Online algorithm visualization tools

## 🧪 Exercises and Challenges

### Basic Exercises
1. Modify binary search to find the first occurrence of a duplicate element
2. Implement DFS to detect cycles in a graph
3. Use BFS to find all nodes at a specific distance from a start node

### Advanced Challenges
1. Implement A* algorithm for the 8-puzzle problem
2. Create a maze solver using DFS and BFS
3. Optimize the 8-puzzle solver with better heuristics

### Extension Projects
1. Visualize search algorithms using matplotlib
2. Compare performance on larger datasets
3. Implement parallel versions of the algorithms

## ⚡ Performance Tips

### For Large Datasets
1. Use binary search on sorted data
2. Consider hash tables for O(1) average search time
3. Implement iterative versions to avoid stack overflow

### For Graph Problems
1. Use adjacency lists for sparse graphs
2. Implement early termination when target is found
3. Consider bidirectional search for pathfinding

### Memory Optimization
1. Use generators for large search spaces
2. Implement depth-limited search variants
3. Clear visited sets when possible

## 🐛 Common Pitfalls

1. **Binary Search**: Forgetting that data must be sorted
2. **DFS**: Stack overflow in deep recursions
3. **BFS**: Memory explosion in wide graphs
4. **8-Puzzle**: Not checking for solvability first

## 📈 Next Steps

After mastering these algorithms, consider exploring:
- Advanced search algorithms (A*, IDA*, beam search)
- Optimization problems (genetic algorithms, simulated annealing)
- Machine learning applications of search
- Real-time and anytime algorithms

## 🤝 Contributing

Feel free to:
- Add more test cases
- Implement visualizations
- Optimize existing code
- Add new search algorithms

## 📄 License

This educational content is provided for learning purposes. Feel free to use and modify for educational goals.

---

**Happy Learning! 🎓**

*Remember: Understanding the theory behind these algorithms is just as important as implementing them. Make sure to read the accompanying report and experiment with different inputs to see how the algorithms behave in various scenarios.* 