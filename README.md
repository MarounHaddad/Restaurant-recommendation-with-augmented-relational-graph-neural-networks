# Restaurant-recommendation-with-augmented-relational-graph-neural-networks

In this project, we use relational graph neural networks for restaurant recommendations. Furthermore, we extract patterns from the interactions of the users and restaurants in order to augment the graph with additional training samples in order to improve the learning.

# Problem definition
The relation between the users and the restaurants is modeled as a heterogeneous bipartite graph. The nodes in the graph belong to two types: Users and Restaurants. The edges that connect the nodes are the reviews written by the users about the restaurants. The number of stars granted by the user in the review is treated as a type of edge in the heterogeneous graph. The problem of recommendation is then treated as a task of semi-supervised edge labeling on the heterogenous bipartite graph. In this case, the model is trained on a subset of labeled edges in order to properly label another subset. The labels are the type of edge in the graph (i.e. the number of stars the user granted to the restaurant).
