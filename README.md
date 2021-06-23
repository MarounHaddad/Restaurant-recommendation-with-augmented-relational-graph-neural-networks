# Restaurant recommendation with augmented relational graph neural networks

In this project, we use relational graph neural networks for restaurant recommendations. Furthermore, we extract patterns from the interactions of the users and restaurants in order to augment the graph with additional training samples in order to improve the learning.

## Problem definition
In this work, we model the relationship between the users and the restaurants as a heterogeneous bipartite graph, Figure 1. The nodes in the graph belong to two types: Users and Restaurants. The edges that connect the nodes are the reviews written by the users about the restaurants. The number of stars granted by the user in the review is represented as a type of edge in the heterogeneous graph. The problem of recommendation is then treated as a task of semi-supervised edge labeling on the heterogenous bipartite graph. In this case, the model is trained on a subset of labeled edges in order to properly label another test subset. The labels to be classified are the type of edge in the graph (i.e. the number of stars the user granted to the restaurant). By properly predicting the label on the edge or the number of stars this user would grant this restaurant, we could effectively recommend or not recommend the restaurant for the user.

<p align="center">
  <img width="400" height="300" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/bipartite%20heterogenous%20graph.png">
</p>
 <p align="center"><em>Figure 1 - Heterogeneous Bipartite Graph that models the number of stars granted by the user to the restaurant.</em></p>
 
## Augmenting the graph with FP-Growth
One of the problems that we encounter when training a model on real-life graphs is "sparsity", where only a small subset of the nodes in the graph are connected to each other. Therefore, in the case of edge labeling or prediction, we do not have enough samples to properly train the model. One way of handling this issue is by enlarging the size of the graph (i.e. adding more nodes). However, adding new nodes would increase the complexity of the training without necessarily adding new training examples to the already existing nodes. In order to mitigate this issue, we can augment the graph with new edges. However, these edges cannot be added in a random way and would have to respect the underlying tendencies in the data. Therefore, we propose extracting patterns from the data in the form of association rules that can be used as potential relationships in the graph in order to efficiently enlarge the training sample.  

In order to implement the proposed method for our case study, first we filter all the training edges by type (number of stars). We consider the restaurants granted the filtered number of stars as our basket. We apply FP-Growth on the filtered data and take all the frequent rules of size 2. For every user, we verify if he is already connected to one of the two restaurants in the rule, we connect him to the other restaurant with the same edge type as the number of stars filtered at the start. This process is repeated for all 5 types of edges (number of stars). Figure 2 summerizes the augmentation process. User John has already granted the restaurant Belon 4 stars. Since Belon is frequently visited by the same users that visit Milos restaurant and grant them both 4 stars. We connect John to Milos restaurant with 4 stars

<p align="center">
  <img width="400" height="300" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/bipartite%20heterogenous%20graph.png">
</p>
 <p align="center"><em>Figure 2 - Data augmentation example for the 4 stars edges.</em></p>
