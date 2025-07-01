"""
Week 2: Graph Search Algorithms - DFS and BFS Implementation
Author: AI Learning Project
Date: 2024

This module implements Depth-First Search (DFS) and Breadth-First Search (BFS)
algorithms using adjacency list representation of graphs.
"""

from collections import deque, defaultdict

class Graph:
    """
    Graph class using adjacency list representation
    """
    
    def __init__(self):
        """Initialize an empty graph"""
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v, directed=False):
        """
        Add an edge between vertices u and v
        
        Args:
            u: Source vertex
            v: Destination vertex
            directed: If False, adds edge in both directions (undirected graph)
        """
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)
        
        if not directed:
            self.graph[v].append(u)
    
    def add_vertex(self, vertex):
        """Add a vertex to the graph"""
        self.vertices.add(vertex)
        if vertex not in self.graph:
            self.graph[vertex] = []
    
    def display_graph(self):
        """Display the adjacency list representation"""
        print("Graph Adjacency List:")
        for vertex in sorted(self.vertices):
            neighbors = sorted(self.graph[vertex])
            print(f"  {vertex} -> {neighbors}")
    
    def dfs(self, start_vertex, target=None):
        """
        Depth-First Search (DFS) - Iterative Implementation
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Args:
            start_vertex: Starting vertex for search
            target: Target vertex to search for (optional)
        
        Returns:
            list: Path of vertices visited during DFS
            bool: True if target found (when target specified)
        """
        if start_vertex not in self.vertices:
            print(f"Start vertex {start_vertex} not in graph")
            return [] if target is None else False
        
        visited = set()
        stack = [start_vertex]
        path = []
        
        print(f"DFS starting from vertex {start_vertex}")
        
        while stack:
            vertex = stack.pop()
            
            if vertex not in visited:
                visited.add(vertex)
                path.append(vertex)
                print(f"  Visiting: {vertex}")
                
                if target and vertex == target:
                    print(f"  Target {target} found!")
                    return True
                
                # Add neighbors to stack (in reverse order to maintain left-to-right traversal)
                neighbors = sorted(self.graph[vertex], reverse=True)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        if target:
            print(f"  Target {target} not found")
            return False
        
        return path
    
    def dfs_recursive(self, start_vertex, visited=None, path=None, target=None):
        """
        Depth-First Search (DFS) - Recursive Implementation
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Args:
            start_vertex: Starting vertex for search
            visited: Set of visited vertices (for recursion)
            path: List of vertices in path (for recursion)
            target: Target vertex to search for (optional)
        
        Returns:
            list: Path of vertices visited during DFS
            bool: True if target found (when target specified)
        """
        if visited is None:
            visited = set()
            path = []
            print(f"DFS (Recursive) starting from vertex {start_vertex}")
        
        if start_vertex not in self.vertices:
            print(f"Start vertex {start_vertex} not in graph")
            return [] if target is None else False
        
        visited.add(start_vertex)
        path.append(start_vertex)
        print(f"  Visiting: {start_vertex}")
        
        if target and start_vertex == target:
            print(f"  Target {target} found!")
            return True
        
        # Visit neighbors
        neighbors = sorted(self.graph[start_vertex])
        for neighbor in neighbors:
            if neighbor not in visited:
                result = self.dfs_recursive(neighbor, visited, path, target)
                if target and result:
                    return True
        
        if target:
            return False
        
        return path
    
    def bfs(self, start_vertex, target=None):
        """
        Breadth-First Search (BFS) Implementation
        Time Complexity: O(V + E)
        Space Complexity: O(V)
        
        Args:
            start_vertex: Starting vertex for search
            target: Target vertex to search for (optional)
        
        Returns:
            list: Path of vertices visited during BFS
            bool: True if target found (when target specified)
        """
        if start_vertex not in self.vertices:
            print(f"Start vertex {start_vertex} not in graph")
            return [] if target is None else False
        
        visited = set()
        queue = deque([start_vertex])
        path = []
        
        print(f"BFS starting from vertex {start_vertex}")
        
        while queue:
            vertex = queue.popleft()
            
            if vertex not in visited:
                visited.add(vertex)
                path.append(vertex)
                print(f"  Visiting: {vertex}")
                
                if target and vertex == target:
                    print(f"  Target {target} found!")
                    return True
                
                # Add neighbors to queue
                neighbors = sorted(self.graph[vertex])
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
        
        if target:
            print(f"  Target {target} not found")
            return False
        
        return path
    
    def shortest_path_bfs(self, start, end):
        """
        Find shortest path between two vertices using BFS
        
        Args:
            start: Starting vertex
            end: Ending vertex
        
        Returns:
            list: Shortest path from start to end, empty if no path exists
        """
        if start not in self.vertices or end not in self.vertices:
            return []
        
        visited = set()
        queue = deque([(start, [start])])
        
        while queue:
            vertex, path = queue.popleft()
            
            if vertex == end:
                return path
            
            if vertex not in visited:
                visited.add(vertex)
                
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found

def create_sample_graph():
    """Create a sample graph for demonstration"""
    g = Graph()
    
    # Add edges to create a connected graph
    edges = [
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'F'),
        ('D', 'G'),
        ('E', 'G'), ('E', 'H'),
        ('F', 'H'),
        ('G', 'I'),
        ('H', 'I')
    ]
    
    for u, v in edges:
        g.add_edge(u, v)
    
    return g

def create_directed_graph():
    """Create a sample directed graph"""
    g = Graph()
    
    # Add directed edges
    edges = [
        ('1', '2'), ('1', '3'),
        ('2', '4'),
        ('3', '4'), ('3', '5'),
        ('4', '6'),
        ('5', '6')
    ]
    
    for u, v in edges:
        g.add_edge(u, v, directed=True)
    
    return g

def demonstrate_algorithms():
    """Demonstrate DFS and BFS algorithms"""
    print("=== Graph Search Algorithms Demonstration ===\n")
    
    # Create and display sample graph
    print("1. Undirected Graph Example:")
    g1 = create_sample_graph()
    g1.display_graph()
    
    print("\n--- DFS Traversal ---")
    dfs_path = g1.dfs('A')
    print(f"DFS Path: {' -> '.join(dfs_path)}")
    
    print("\n--- DFS Recursive Traversal ---")
    dfs_recursive_path = g1.dfs_recursive('A')
    print(f"DFS Recursive Path: {' -> '.join(dfs_recursive_path)}")
    
    print("\n--- BFS Traversal ---")
    bfs_path = g1.bfs('A')
    print(f"BFS Path: {' -> '.join(bfs_path)}")
    
    print("\n--- Search for Specific Target ---")
    target = 'G'
    print(f"Searching for vertex '{target}':")
    dfs_found = g1.dfs('A', target)
    bfs_found = g1.bfs('A', target)
    
    print(f"\n--- Shortest Path (BFS) ---")
    shortest = g1.shortest_path_bfs('A', 'I')
    print(f"Shortest path from A to I: {' -> '.join(shortest)}")
    
    print("\n" + "="*50)
    print("2. Directed Graph Example:")
    g2 = create_directed_graph()
    g2.display_graph()
    
    print("\n--- DFS on Directed Graph ---")
    dfs_directed = g2.dfs('1')
    print(f"DFS Path: {' -> '.join(dfs_directed)}")
    
    print("\n--- BFS on Directed Graph ---")
    bfs_directed = g2.bfs('1')
    print(f"BFS Path: {' -> '.join(bfs_directed)}")

def compare_dfs_bfs():
    """Compare DFS and BFS characteristics"""
    print("\n=== DFS vs BFS Comparison ===")
    print("Depth-First Search (DFS):")
    print("  - Uses Stack (LIFO) or Recursion")
    print("  - Goes deep into graph before exploring siblings")
    print("  - Memory efficient for deep graphs")
    print("  - May not find shortest path")
    print("  - Good for: Topological sorting, cycle detection, pathfinding in mazes")
    
    print("\nBreadth-First Search (BFS):")
    print("  - Uses Queue (FIFO)")
    print("  - Explores all neighbors before going deeper")
    print("  - Finds shortest path in unweighted graphs")
    print("  - More memory intensive")
    print("  - Good for: Shortest path, minimum spanning tree, social networks")

if __name__ == "__main__":
    print("Week 2: Graph Search Algorithms - DFS and BFS\n")
    
    demonstrate_algorithms()
    compare_dfs_bfs()
    
    print("\n=== Key Concepts ===")
    print("1. Graph representation using adjacency lists")
    print("2. DFS: Stack-based, goes deep first")
    print("3. BFS: Queue-based, explores level by level")
    print("4. Both have O(V + E) time complexity")
    print("5. BFS guarantees shortest path in unweighted graphs") 