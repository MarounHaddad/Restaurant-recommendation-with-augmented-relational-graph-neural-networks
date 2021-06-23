"""
This file contains the functions that are used to visualize the embeddings
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE


def visualize_edge(embedding, stars):
    """
    visualize one large plot for a model and
    color it according to certain feature labels
    :param embedding: the edge embeddings
    :param stars: the edge stars (class)
    """
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embedding)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    area = np.pi * 3
    # dot_colors = list(mcolors.CSS4_COLORS)
    # random.shuffle(dot_colors)
    dot_colors = ["blue", "red", "orange", "green", "yellow", "cyan", "purple", "black", "pink"]

    plot_secondary_index = 0
    number_classes = len(set(stars))
    xc = []
    yc = []
    for c in range(0, number_classes):
        xc.append([])
        yc.append([])
        for i in range(0, len(embedding)):
            if stars[i] == c:
                xc[c].append(x[i])
                yc[c].append(y[i])
        plt.scatter(xc[c], yc[c], s=area, c=dot_colors[c], alpha=0.5)
    plot_secondary_index += 1
    plt.show()


def visualize_graph(test_edges, embedding):
    """
    visualize the graph according to the node embeddings
    :param test_edges: the list of test edges
    :param embedding: the node embeddings
    """
    graph = nx.Graph()
    colors = []

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embedding)
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]

    pos = {}

    for edge in test_edges:
        nodes = edge.split("_")
        if graph.has_node(nodes[0]):
            continue
        graph.add_node(nodes[0])
        colors.append('blue')
        node_id = int(nodes[0])
        pos[nodes[0]] = [x[node_id], y[node_id]]
        # pos[nodes[0]] = embedding[node_id]

    for edge in test_edges:
        nodes = edge.split("_")
        if graph.has_node(nodes[1]):
            continue
        graph.add_node(nodes[1])
        colors.append('red')
        node_id = int(nodes[1])
        pos[nodes[1]] = [x[node_id], y[node_id]]
        # pos[nodes[1]] = embedding[node_id]

    for edge in test_edges:
        nodes = edge.split("_")
        graph.add_edge(nodes[0], nodes[1])

    nx.draw_networkx(graph, pos, node_color=colors, with_labels=True, node_size=300)
    plt.show()
