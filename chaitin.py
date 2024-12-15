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

# Define the edge case as a graph (adjacency list representation)
edge_case_graph = {
    0: [1, 2, 3],
    1: [0, 2, 4],
    2: [0, 1, 5],
    3: [0, 4, 5],
    4: [1, 3, 5],
    5: [2, 3, 4]
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