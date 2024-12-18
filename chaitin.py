import time
import networkx as nx
import matplotlib.pyplot as plt

def chaitin_map_coloring(graph):
    def calculate_degrees():
        return {state: len(neighbors) for state, neighbors in graph.items()}

    def find_removable_state(degrees):
        return min(degrees, key=degrees.get) if degrees else None

    def color_state(state):
        used_colors = set(coloring[neighbor] for neighbor in graph[state] if neighbor in coloring)
        available_colors = set(colors) - used_colors
        return available_colors.pop() if available_colors else None

    colors = ['red', 'green', 'blue']
    coloring = {}
    stack = []

    # Step 1: Simplify
    degrees = calculate_degrees()
    while degrees:
        removable = find_removable_state(degrees)
        stack.append(removable)
        for neighbor in graph[removable]:
            if neighbor in degrees:
                degrees[neighbor] -= 1
        del degrees[removable]

    # Step 2: Select
    while stack:
        state = stack.pop()
        color = color_state(state)
        if color is None:
            return None  # Coloring not possible
        coloring[state] = color

    return coloring

# Edge Case 1-5: Chaitin's fails on 3 colors, needs 4
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

# Graph Case 1-5: Chaitin's only needs 3 colors
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

def visualize_colored_graph(graph, coloring):
    G = nx.Graph(graph)
    pos = nx.spring_layout(G)
    
    plt.figure(figsize=(15, 10))
    
    # Draw nodes
    for color in set(coloring.values()):
        node_list = [node for node in G.nodes() if coloring[node] == color]
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=color, node_size=500)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Graph Colored Using Chaitin's Algorithm")
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('chaitin_graph_colored.png', dpi=300, bbox_inches='tight')
    print("Graph visualization saved as 'chaitin_graph_colored.png'")

# Run the solver
start_time = time.time()

# edge_case_graph = graph5
result = chaitin_map_coloring(edge_case_graph)

end_time = time.time()
execution_time = end_time - start_time

if result:
    print("Colored Graph:")
    for state, color in result.items():
        print(f"{state}: {color}")
    
    # Visualize the colored graph
    visualize_colored_graph(edge_case_graph, result)
else:
    print("Coloring not possible with the given colors")

print(f"\nExecution Time: {execution_time:.6f} seconds")

# Verify the coloring
if result:
    is_valid = all(result[s] != result[n] for s in edge_case_graph for n in edge_case_graph[s])
    print(f"\nIs the coloring valid? {'Yes' if is_valid else 'No'}")