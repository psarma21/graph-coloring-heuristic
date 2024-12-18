from typing import List, Set, Dict, Tuple
from collections import defaultdict
import heapq
import networkx as nx
import matplotlib.pyplot as plt
import time

class PSHSolver:
    def __init__(self, graph: Dict[int, List[int]], beam_width: int = 3, max_backtracks: int = 100):
        self.graph = graph
        self.beam_width = beam_width
        self.colors = [0, 1, 2]  # Representing Red, Green, Blue
        self.max_backtracks = max_backtracks
        
    def penalty_score(self, partial_coloring: Dict[int, int]) -> float:
        """
        Calculate penalty score for a partial coloring with much stronger penalties for conflicts
        """
        penalty = 0.0
        
        for node, color in partial_coloring.items():
            colored_neighbors = 0
            conflicts = 0
            
            for neighbor in self.graph[node]:
                if neighbor in partial_coloring:
                    colored_neighbors += 1
                    if partial_coloring[neighbor] == color:
                        conflicts += 1000.0
                        
            penalty += conflicts
            
            if colored_neighbors < len(self.graph[node]):
                penalty += 0.1 * (len(self.graph[node]) - colored_neighbors)
                
        return penalty
    
    def is_safe_color(self, node: int, color: int, coloring: Dict[int, int]) -> bool:
        """Check if it's safe to color the node with given color"""
        for neighbor in self.graph[node]:
            if neighbor in coloring and coloring[neighbor] == color:
                return False
        return True
    
    def backtrack_search(self, partial_coloring: Dict[int, int], uncolored: Set[int], 
                        backtracks_remaining: int) -> Dict[int, int]:
        """Attempt to complete partial coloring using backtracking"""
        if not uncolored:
            if all(self.is_safe_color(node, partial_coloring[node], {k:v for k,v in partial_coloring.items() if k != node})
                  for node in partial_coloring):
                return partial_coloring
            return None
        
        if backtracks_remaining <= 0:
            return None
            
        next_node = max(uncolored, 
                       key=lambda n: sum(1 for nb in self.graph[n] if nb in partial_coloring))
        
        for color in self.colors:
            if self.is_safe_color(next_node, color, partial_coloring):
                new_coloring = partial_coloring.copy()
                new_coloring[next_node] = color
                new_uncolored = uncolored - {next_node}
                
                result = self.backtrack_search(new_coloring, new_uncolored, 
                                            backtracks_remaining - 1)
                if result:
                    return result
                
        return None
    
    def solve(self) -> Dict[int, int]:
        """
        Solve using beam search with backtracking fallback
        """
        nodes = set(self.graph.keys())
        initial_coloring = {}
        
        start_node = max(nodes, key=lambda n: len(self.graph[n]))
        initial_coloring[start_node] = 0
        
        beam = [(0, initial_coloring)]
        uncolored = nodes - {start_node}
        backtracks = self.max_backtracks
        
        while uncolored and beam and backtracks > 0:
            new_beam = []
            
            for _, coloring in beam:
                backtrack_solution = self.backtrack_search(
                    coloring, uncolored, min(10, backtracks))
                if backtrack_solution and is_valid_coloring(self.graph, backtrack_solution):
                    return backtrack_solution
                
                next_node = max(uncolored, 
                              key=lambda n: sum(1 for nb in self.graph[n] if nb in coloring))
                
                for color in self.colors:
                    if self.is_safe_color(next_node, color, coloring):
                        new_coloring = coloring.copy()
                        new_coloring[next_node] = color
                        penalty = self.penalty_score(new_coloring)
                        new_beam.append((penalty, new_coloring))
                
                backtracks -= 1
            
            new_beam = [(p, c) for p, c in new_beam if p < 500]
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:self.beam_width]
            
            if beam:
                best_coloring = beam[0][1]
                uncolored = nodes - set(best_coloring.keys())
        
        if uncolored:
            return self.backtrack_search({start_node: 0}, nodes - {start_node}, 
                                      self.max_backtracks)
        
        return beam[0][1] if beam else None

def is_valid_coloring(graph: Dict[int, List[int]], coloring: Dict[int, int]) -> bool:
    """Verify if the coloring is valid"""
    for node in graph:
        for neighbor in graph[node]:
            if coloring[node] == coloring[neighbor]:
                return False
    return True

def color_to_name(color: int) -> str:
    return ['Red', 'Green', 'Blue'][color]

def visualize_graph(graph: Dict[int, List[int]], coloring: Dict[int, int], filename: str):
    """Create and save a visualization of the colored graph"""
    # Create NetworkX graph
    G = nx.Graph(graph)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create a layout for the nodes
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Define the actual colors for visualization
    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    node_colors = [color_map[coloring[node]] for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, 
           node_color=node_colors,
           with_labels=True,
           node_size=1500,
           font_size=16,
           font_weight='bold',
           font_color='white',
           edge_color='gray',
           width=2)
    
    # Add a title
    plt.title("Graph 3-Coloring Visualization", pad=20, size=16)
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=name, markersize=15)
                      for color, name in zip(['red', 'green', 'blue'], 
                                          ['Red (0)', 'Green (1)', 'Blue (2)'])]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Test Cases 1-5: PSH only needs 3 (Chaitin's needed 4)
edge_case_graph1 = {
    0: [1, 2, 3],
    1: [0, 2, 4],
    2: [0, 1, 5],
    3: [0, 4, 5],
    4: [1, 3, 5],
    5: [2, 3, 4]
}

edge_case_graph2 = {
    0: [1,2,3],
    1: [0,2,4],
    2: [0,1,5],
    3: [0,4,5],
    4: [1,3,5],
    5: [2,3,4,6],
    6: [5,7,8,9],
    7: [6,8,10],
    8: [6,7,11],
    9: [6,10,11],
    10: [7,9,11],
    11: [8,9,10]
}

edge_case_graph3 = {
    # First Prism (0–5)
    0: [1, 2, 3],
    1: [0, 2, 4],
    2: [0, 1, 5],
    3: [0, 4, 5],
    4: [1, 3, 5],
    5: [2, 3, 4, 6],   # Connects to second prism

    # Second Prism (6–11)
    6: [5, 7, 8, 9],
    7: [6, 8, 10],
    8: [6, 7, 11],
    9: [6, 10, 11],
    10:[7, 9, 11],
    11:[8, 9, 10, 12], # Connects to third prism

    # Third Prism (12–17)
    12:[11, 13, 14, 15],
    13:[12, 14, 16],
    14:[12, 13, 17],
    15:[12, 16, 17],
    16:[13, 15, 17],
    17:[14, 15, 16]
}

edge_case_graph4 = {
    0: [1,3,5],
    1: [0,2,4],
    2: [1,3,4],
    3: [0,2,5],
    4: [1,2,5],
    5: [0,3,4,6],   # Connection to second house
    
    6: [5,7,9,11],
    7: [6,8,10],
    8: [7,9,10],
    9: [6,8,11],
    10:[7,8,11],
    11:[6,9,10]
}

edge_case_graph5 = {
    # First house structure (0-5)
    0: [1, 2, 3],
    1: [0, 2, 4],
    2: [0, 1, 5],
    3: [0, 4, 5],
    4: [1, 3, 5],
    5: [2, 3, 4, 6],  # Bridge to middle structure
    
    # Middle diamond structure (6-9)
    6: [5, 7, 8],
    7: [6, 8, 9],
    8: [6, 7, 9],
    9: [7, 8, 10],    # Bridge to final structure
    
    # Final house structure (10-15)
    10: [9, 11, 12, 13],
    11: [10, 12, 14],
    12: [10, 11, 15],
    13: [10, 14, 15],
    14: [11, 13, 15],
    15: [12, 13, 14]
}

# Test Cases 1-5: PSH only needs 3 (so did Chaitin's)
graph1 = {
    0: [1,4],
    1: [0,2],
    2: [1,3],
    3: [2,4],
    4: [0,3,5],
    5: [4,6,7],
    6: [5,7],
    7: [5,6]
}

graph2 = {
    0: [1,2,4,5],
    1: [0,2,3,5],
    2: [0,1,3,4],
    3: [1,2,4,5],
    4: [0,2,3,5],
    5: [0,1,3,4,6],  # connection to second half

    6: [5,7,8,10,11],
    7: [6,8,9,11],
    8: [6,7,9,10],
    9: [7,8,10,11],
    10:[6,8,9,11],
    11:[6,7,9,10]
}

graph3 = {
    0: [1,2],
    1: [0,2],
    2: [0,1,3],
    3: [2,4],
    4: [3,5,6],
    5: [4,6],
    6: [4,5]
}

graph4 = {
    0: [1,3],
    1: [0,2,4],
    2: [1,5],
    3: [0,4],
    4: [1,3,5],
    5: [2,4]
}

graph5 = {
    0: [1,5],
    1: [0,2,3],
    2: [1,3,4],
    3: [1,2,4,6],
    4: [2,3,5,6],
    5: [0,4],
    6: [3,4]
}

# Run the solver
solver = PSHSolver(edge_case_graph, beam_width=3, max_backtracks=100)
start_time = time.time()
solution = solver.solve()
end_time = time.time()

execution_time = end_time - start_time

if solution:
    print("Found valid 3-coloring!")
    print(f"\nExecution Time: {execution_time:.6f} seconds")
    print("\nNode colors:")
    for node in sorted(edge_case_graph.keys()):
        print(f"Node {node}: {color_to_name(solution[node])}")
        
    print("\nVerifying solution...")
    if is_valid_coloring(edge_case_graph, solution):
        print("Solution is valid! No adjacent nodes have the same color.")
        # Save visualization
        visualize_graph(edge_case_graph, solution, "graph_coloring.png")
        print("\nVisualization saved as 'graph_coloring.png'")
    else:
        print("Error: Solution is invalid!")
else:
    print("No valid 3-coloring found!")

# Print the graph structure
# print("\nGraph structure:")
# for node in sorted(edge_case_graph.keys()):
#     print(f"Node {node} is connected to: {sorted(edge_case_graph[node])}")