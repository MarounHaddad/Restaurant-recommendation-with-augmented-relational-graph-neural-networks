# Restaurant recommendation with augmented relational graph neural networks

In this project, we use relational graph neural networks for restaurant recommendations. Furthermore, we extract patterns from the interactions of the users and restaurants in order to augment the graph with additional training samples in order to improve the learning.

## Problem definition
In this work, we model the relation between the users and the restaurants as a heterogeneous bipartite graph, figure . The nodes in the graph belong to two types: Users and Restaurants. The edges that connect the nodes are the reviews written by the users about the restaurants. The number of stars granted by the user in the review is represented as a type of edge in the heterogeneous graph. The problem of recommendation is then treated as a task of semi-supervised edge labeling on the heterogenous bipartite graph. In this case, the model is trained on a subset of labeled edges in order to properly label another subset. The target labels are the type of edge in the graph (i.e. the number of stars the user granted to the restaurant).

<p align="center">
  <img width="500" height="400" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/bipartite%20heterogenous%20graph.png">
  <em>image_caption</em>
</p>

