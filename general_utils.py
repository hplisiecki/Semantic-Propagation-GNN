def visualize_scaling_factors(data, activations, layer_index=0, save=None):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    # Set the DPI (dots per inch) for high-resolution output
    fig, ax = plt.subplots(figsize=(8, 6))


    # Extract words and node information from the data
    words = data.word[0]  # List of words associated with token nodes
    num_token_nodes = len(words)  # Number of token nodes (words)
    num_nodes = data.x.shape[0]   # Total number of nodes in the graph

    # Create labels for each node in the graph
    node_labels = []
    for i in range(num_nodes):
        if i < num_token_nodes:
            # Assign word labels to token nodes
            node_labels.append(words[i])
        else:
            # Assign generic labels to sentence nodes or other types
            node_labels.append(f"Sentence_Node_{i - num_token_nodes + 1}")

    # Extract edge indices (source and destination nodes for each edge)
    edge_index = data.edge_index.detach().cpu().numpy()

    # Get scaling factors from the specified layer's activations
    s_ij = activations['s_ij_list'][layer_index]  # Shape: [num_edges, hidden_dim]
    s_ij_avg = s_ij.mean(dim=1).numpy()  # Compute average scaling factor over hidden dimensions

    # Initialize a directed graph using NetworkX
    G = nx.DiGraph()

    # Add nodes to the graph with their corresponding labels
    for idx, label in enumerate(node_labels):
        G.add_node(idx, label=label)

    # Define a custom colormap for edge scaling factors
    colors = ["red", "white", "yellow"]  # Colors represent negative, neutral, and positive scaling factors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    try:
        # Normalize scaling factors around zero for the colormap
        norm = TwoSlopeNorm(vmin=np.min(s_ij_avg), vcenter=0, vmax=np.max(s_ij_avg))
    except:
        # If zero is not between min and max, center around the mean
        mean = np.mean(s_ij_avg)
        norm = TwoSlopeNorm(vmin=np.min(s_ij_avg), vcenter=mean, vmax=np.max(s_ij_avg))

    # Add edges to the graph with attributes for scaling factors and colors
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]    # Source node index
        dest = edge_index[1, i]   # Destination node index
        scaling_factor = s_ij_avg[i]  # Average scaling factor for this edge
        color = cmap(norm(scaling_factor))  # Color based on normalized scaling factor
        G.add_edge(src, dest, scaling_factor=scaling_factor, color=color)

    # Compute positions for nodes using a spring layout for better visualization
    pos = nx.spring_layout(G, seed=42)  # Seed ensures consistent layout across runs

    # Prepare edge attributes for visualization
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]  # List of edge colors
    edge_labels = {(u, v): f"{G[u][v]['scaling_factor']:.2f}" for u, v in G.edges()}  # Edge labels with scaling factors

    # Retrieve attention weights for nodes, if available
    attention_weights = activations.get('attention_weights', None)
    if attention_weights is not None:
        # Convert attention weights to a 1D NumPy array
        attention_weights = attention_weights.squeeze().numpy()
        # Normalize attention weights for coloring nodes
        node_norm = TwoSlopeNorm(vmin=np.min(attention_weights),
                                 vcenter=np.mean(attention_weights),
                                 vmax=np.max(attention_weights))
        # Define a colormap for node colors based on attention weights
        node_cmap = LinearSegmentedColormap.from_list("node_cmap", ["lightgrey", "blue"])
        node_colors = [node_cmap(node_norm(attention_weights[i])) for i in range(num_nodes)]
        # Scale node sizes proportionally to attention weights
        node_sizes = 500 + attention_weights * 5000  # Adjust scaling factor as needed
    else:
        # Default node colors and sizes if attention weights are not provided
        node_colors = ['lightblue' if i < num_token_nodes else 'lightgreen' for i in range(num_nodes)]
        node_sizes = 800  # Uniform node size

    # Draw the graph with nodes and edges
    nx.draw(
        G, pos,  # Graph and node positions
        labels=nx.get_node_attributes(G, 'label'),  # Node labels
        with_labels=True,  # Display labels
        edge_color=edge_colors,  # Edge colors based on scaling factors
        node_color=node_colors,  # Node colors based on attention weights
        node_size=node_sizes,    # Node sizes
        font_size=10,  # Font size for labels
        width=2,       # Edge line width
        arrowsize=15,  # Arrow size for directed edges
        connectionstyle='arc3, rad=0.1'  # Slightly curved edges for better clarity
    )

    # Improve edge label placement along the curved edges
    ax = plt.gca()  # Get the current Axes instance
    for (u, v), label in edge_labels.items():
        # Get start and end positions of the edge
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        # Calculate control points for the Bezier curve (used for curved edges)
        rad = 0.1  # Curvature radius, matching the 'rad' parameter in connectionstyle
        ctrl_x = (x0 + x1) / 2 + rad * (y1 - y0)  # Control point x-coordinate
        ctrl_y = (y0 + y1) / 2 + rad * (x0 - x1)  # Control point y-coordinate

        # Parameter t determines the position along the Bezier curve (0 <= t <= 1)
        t = 0.5  # Midpoint of the curve

        # Calculate the position along the Bezier curve for label placement
        bezier_x = (1 - t)**2 * x0 + 2 * (1 - t) * t * ctrl_x + t**2 * x1
        bezier_y = (1 - t)**2 * y0 + 2 * (1 - t) * t * ctrl_y + t**2 * y1

        # Add the edge label at the calculated position
        ax.text(bezier_x, bezier_y, label,
                fontsize=7, color='black', ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', pad=0.1))

    # Add a colorbar for edge scaling factors
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm)
    cbar.set_label('Edge Scaling Factor')  # Label for the colorbar

    if attention_weights is not None:
        # Add a separate colorbar for node attention weights
        sm_node = ScalarMappable(norm=node_norm, cmap=node_cmap)
        sm_node.set_array([])
        cbar_node = plt.colorbar(sm_node, orientation='horizontal', pad=0.02)
        cbar_node.set_label('Node Attention Weight')  # Label for the node colorbar

    # Set the title of the plot indicating the layer being visualized
    plt.title(f"Scaling Factors Visualization for Layer {layer_index}")
    plt.axis('off')    # Hide the axes for a cleaner look
    plt.tight_layout() # Adjust the padding between and around subplots

    if save:
        # Save the figure to the specified file path
        plt.savefig(save)
    plt.show()  # Display the plot on the screen


def visualize_scaling_factors_simple(data, activations, layer_index=0, save=None, seed = 42, title = None, x_off =0, y_off = 0):
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np

    # Set the DPI (dots per inch) for high-resolution output and define the figure size
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract words and node information from the data
    words = data.word[0]  # List of words associated with token nodes
    num_token_nodes = len(words)  # Number of token nodes (words)
    num_nodes = data.x.shape[0]   # Total number of nodes in the graph

    # Create labels for each node in the graph
    node_labels = []
    for i in range(num_nodes):
        if i < num_token_nodes:
            # Assign word labels to token nodes
            node_labels.append(words[i])
        else:
            # Assign generic labels to sentence nodes or other types
            node_labels.append("S Node")

    # Extract edge indices (source and destination nodes for each edge)
    edge_index = data.edge_index.detach().cpu().numpy()

    # Get scaling factors from the specified layer's activations
    s_ij = activations['s_ij_list'][layer_index]  # Shape: [num_edges, hidden_dim]
    s_ij_avg = s_ij.mean(dim=1).numpy()  # Compute average scaling factor over hidden dimensions

    # Initialize a directed graph using NetworkX
    G = nx.DiGraph()

    # Add nodes to the graph with their corresponding labels
    for idx, label in enumerate(node_labels):
        G.add_node(idx, label=label)

    # Add edges to the graph with attributes for scaling factors
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]    # Source node index
        dest = edge_index[1, i]   # Destination node index
        scaling_factor = s_ij_avg[i]  # Average scaling factor for this edge
        G.add_edge(src, dest, scaling_factor=scaling_factor)

    # Compute positions for nodes using a spring layout for better visualization
    pos = nx.spring_layout(G, seed=seed, k=0.5)  # Seed ensures consistent layout across runs

    # Prepare edge labels with scaling factors
    edge_labels = {(u, v): f"{G[u][v]['scaling_factor']:.2f}" for u, v in G.edges()}  # Edge labels with scaling factors

    # Retrieve attention weights for nodes, if available
    attention_weights = activations.get('attention_weights', None)
    if attention_weights is not None:
        # Convert attention weights to a 1D NumPy array
        attention_weights = attention_weights.squeeze().numpy()
        # Scale node sizes proportionally to attention weights
        node_sizes = 500 + attention_weights * 5000  # Adjust scaling factor as needed
    else:
        # Default node sizes if attention weights are not provided
        node_sizes = 800  # Uniform node size

    # Draw the graph with nodes and edges
    nx.draw(
        G, pos,  # Graph and node positions
        labels=nx.get_node_attributes(G, 'label'),  # Node labels
        with_labels=True,  # Display labels
        node_size=node_sizes,  # Node sizes
        node_color='#4682B4',  # Light green nodes
        font_size=12,  # Font size for labels
        font_color='black',  # Black font color for labels
        edge_color='black',  # Blue edges
        width=2,       # Edge line width
        arrowsize=15,  # Arrow size for directed edges
        connectionstyle='arc3, rad=0.1'  # Slightly curved edges for better clarity
    )
    # Improve edge label placement along the curved edges
    ax = plt.gca()  # Get the current Axes instance
    # x_offset = x_off # Vertical offset for labels, increase this value to move labels higher
    # y_offset = y_off  # Horizontal offset for labels, increase this value to move labels to the right
    # for (u, v), label in edge_labels.items():
    #     # Get start and end positions of the edge
    #     x0, y0 = pos[u]
    #     x1, y1 = pos[v]
    #
    #     # Calculate control points for the Bezier curve (used for curved edges)
    #     rad = 0.1  # Curvature radius, matching the 'rad' parameter in connectionstyle
    #     ctrl_x = (x0 + x1) / 2 + rad * (y1 - y0)  # Control point x-coordinate
    #     ctrl_y = (y0 + y1) / 2 + rad * (x0 - x1)  # Control point y-coordinate
    #
    #     # Parameter t determines the position along the Bezier curve (0 <= t <= 1)
    #     t = 0.5  # Midpoint of the curve
    #
    #     # Calculate the position along the Bezier curve for label placement
    #     bezier_x = (1 - t)**2 * x0 + 2 * (1 - t) * t * ctrl_x + t**2 * x1
    #     bezier_y = (1 - t)**2 * y0 + 2 * (1 - t) * t * ctrl_y + t**2 * y1
    #
    #     adjusted_y = bezier_y + y_offset  # Move label up by the offset
    #     adjusted_x = bezier_x + x_offset  # Move label right by the offset
    #
    #     # Add the edge label at the calculated position
    #     ax.text(adjusted_x, adjusted_y, label,
    #             fontsize=10, color='black', ha='center', va='center',
    #             bbox=dict(facecolor='white', edgecolor='none', pad=0.1))

    # Set the title of the plot indicating the layer being visualized
    if title:
        plt.title(title)
    plt.axis('off')  # Hide the axes for a cleaner look
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust the subplot margins
    plt.tight_layout()  # Adjust the layout to make sure everything fits without overlap

    if save:
        # Save the figure to the specified file path with high DPI
        plt.savefig(save, dpi=1200)
    plt.show()  # Display the plot on the screen
