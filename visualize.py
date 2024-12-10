import networkx as nx
import matplotlib.pyplot as plt
from netgraph import Graph
from sklearn.manifold import TSNE
from matplotlib.patches import FancyArrowPatch, ArrowStyle



# from torch_geometric.utils import to_networkx

# G = to_networkx(data, to_undirected=True)
# visualize_graph(G, color=data.y)
# import networkx as nx
# import torch

def create_nx_multigraph(n_id, e_id, edge_index):
    """
    Creates a NetworkX multigraph from the given tensors.

    Args:
    - n_id (torch.Tensor): Tensor containing node IDs.
    - e_id (torch.Tensor): Tensor containing edge IDs (optional for edge attributes).
    - edge_index (torch.Tensor): Tensor containing edge pairs in COO format.

    Returns:
    - G (nx.MultiGraph): The resulting NetworkX multigraph.
    """
    # Convert tensors to CPU and numpy for easier handling with NetworkX
    n_id = n_id.cpu().numpy()
    e_id = e_id.cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    
    # Create an empty NetworkX multigraph
    G = nx.MultiDiGraph()  # Use nx.MultiDiGraph() for a directed multigraph

    # Add nodes
    G.add_nodes_from(n_id)
    
    # Add edges with unique keys (based on index in e_id)
    for idx, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
        G.add_edge(n_id[src], n_id[dst], key=idx, edge_id=e_id[idx])  # key ensures uniqueness

    # For standard graphs
    # print("Edges:", list(G.edges(data=True)))

    # For multigraphs (include keys for unique edge identification)
    # print("Edges (with keys):", list(G.edges(data=True, keys=True)))

    return G


def create_nx_graph(n_id, e_id, edge_index, directed=False):
    # Convert tensors to CPU and numpy for easier handling with NetworkX
    n_id = n_id.cpu().numpy()
    e_id = e_id.cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    
    # Create an empty NetworkX graph
    # G = nx.Graph()
    if directed:
        G = nx.DiGraph()  # For directed graphs
    else:
        G = nx.Graph()  # For undirected graphs
    
    # Add nodes
    G.add_nodes_from(n_id)
    
    # Add edges with optional edge IDs as attributes
    for idx, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
        # print(n_id[src], n_id[dst])
        G.add_edge(n_id[src], n_id[dst], edge_id=e_id[idx])
    
    return G


def visualize_multigraph_netgraph(G, directed=True):
    Graph(G, arrows = directed, edge_layout ='curved')
    plt.title("Multigraph Visualization")
    plt.savefig("plots/multigraph_plot.png")



def visualize_multigraph(G):
    """
    Visualizes a multigraph with multiple edges.
    Args:
        G (nx.MultiGraph or nx.MultiDiGraph): The multigraph to visualize.
    """
    # plt.figure(figsize=(7, 7))
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # Generate positions for nodes
    # pos = nx.spring_layout(G, seed=21, k=0.8, scale=2)
    # pos = nx.spectral_layout(G)
    pos = nx.circular_layout(G)
    # n = len(G.nodes)
    # rows = int(n**0.5) + 1
    # pos = {node: (i % rows, i // rows) for i, node in enumerate(G.nodes)}
    

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    
    # Draw edges, ensuring multigraph edges are visualized properly
    # for edge in G.edges(keys=True):
    #     nx.draw_networkx_edges(G, pos, edgelist=[edge], alpha=0.7, edge_color='gray')

    ec = {}

    for i, (u, v, key) in enumerate(G.edges(keys=True)):
        loc = str(u)+"-"+str(v)
        if loc in ec:
            ec[loc] += 1
        else:
            ec[loc] = 0.5
        ang =  -1
        if ec[loc]%2 == 1:
            ang = 1
        ang =  ((ec[loc]/2)/15)*ang
        # rad = rad_values[i % len(rad_values)]
        draw_curved_edge(ax, pos, u, v, ang)
        # Optionally, add node labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

    plt.title("Multigraph Visualization")
    plt.savefig("plots/multigraph_plot.png")
    print("Plot saved as 'plots/multigraph_plot.png'")

def visualize_graph(G):
    # nx.draw(G, with_labels=True, node_color='blue', edge_color='gray')
   
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color='lightblue', edge_color='black')
    plt.title("Neighbor Plot")
    plt.savefig("plots/plot.png") 



def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()



def draw_curved_edge(ax, pos, p1, p2, rad):
    """
    Draws a curved edge between two points.

    Args:
        ax: Matplotlib axis to draw on.
        pos: Node positions.
        p1: Start node.
        p2: End node.
        rad: Curvature radius (positive for upward, negative for downward).
    """
    src = pos[p1]
    dst = pos[p2]
    if p1<p2:
        color = 'blue'
    else:
        color = 'green'
    # Create a curved edge with FancyArrowPatch
    edge = FancyArrowPatch(
        src, dst,
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        alpha=0.7,
        linewidth=2,
        arrowstyle=ArrowStyle("-")
    )
    ax.add_patch(edge)

def visualize_with_curved_edges(G, rad_values):
    """
    Visualizes a graph with curved edges.

    Args:
        G (nx.Graph or nx.MultiGraph): The graph to visualize.
        rad_values (list): List of curvature values for the edges.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Generate positions for the nodes
    pos = nx.spring_layout(G, seed=42, scale=2)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", ax=ax)

    # Draw curved edges
    for i, (u, v, key) in enumerate(G.edges(keys=True)):
        rad = rad_values[i % len(rad_values)]
        draw_curved_edge(ax, pos, u, v, rad)

    plt.title("Graph with Curved Edges")
    plt.axis("off")
    plt.savefig("plots/plot2.png") 


