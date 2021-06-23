# Restaurant recommendation with augmented relational graph neural networks

In this project, we use relational graph neural networks (rGCN) for restaurant recommendations on heterogeneous bipartite graphs. Furthermore, we extract patterns or associations from the interactions of the users and restaurants in order to augment the graph with additional training samples for the goal of improving the learning of the model.  

## Problem definition
In this work, we model the relationship between the users and the restaurants as a heterogeneous bipartite graph, Figure 1. The nodes in the graph belong to two types: Users and Restaurants. The edges that connect the nodes are the reviews written by the users about the restaurants. The number of stars granted by the user in the review is represented as a type of edge in the heterogeneous graph. The problem of recommendation is then treated as a task of semi-supervised edge labeling on the heterogenous bipartite graph. In this case, the model is trained on a subset of labeled edges in order to properly label another test subset. The labels to be classified are the type of edge in the graph (i.e. the number of stars the user granted to the restaurant). By properly predicting the label on the edge or the number of stars this user would grant this restaurant, we could effectively recommend or not recommend the restaurant for the user.

<p align="center">
  <img width="400" height="300" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/bipartite%20heterogenous%20graph.png">
</p>
 <p align="center"><em>Figure 1 - Heterogeneous Bipartite Graph that models the number of stars granted by the user to the restaurant.</em></p>
 
## Augmenting the graph with FP-Growth
One of the problems that we encounter when training a model on real-life graphs is "sparsity", where only a small subset of the nodes in the graph are connected to each other. Therefore, in the case of edge labeling or prediction, we do not have enough samples to properly train the model. One way of handling this issue is by enlarging the size of the graph (i.e. adding more nodes). However, adding new nodes would increase the complexity of the training without necessarily adding new training examples to the already existing nodes. In order to mitigate this issue, we can augment the graph with new edges. However, these edges cannot be added in a random way and would have to respect the underlying tendencies in the data. Therefore, we propose extracting patterns from the data in the form of association rules that can be used as potential relationships in the graph in order to efficiently enlarge the training sample.  

In order to implement the proposed method for our case study, first, we filter all the training edges by type (number of stars). We consider the restaurants granted the filtered number of stars as our basket. We apply FP-Growth [1] on the filtered data and take all the frequent rules of size 2. For every user, we verify if he is already connected to one of the two restaurants in the rule, we connect him to the other restaurant with the same edge type as the number of stars filtered at the start. This process is repeated for all 5 types of edges (number of stars). Figure 2 summarizes the augmentation process. User John has already granted the restaurant Belon 4 stars. Since Belon is frequently visited by the same users that visit Milos restaurant and these users tend to grant them both 4 stars, we connect John to Milos restaurant with 4 stars.

<p align="center">
  <img width="300" height="400" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/data%20augementation.png">
</p>
 <p align="center"><em>Figure 2 - Data augmentation example for the 4 stars edges.</em></p>
 
Note: Another way to perform the augmentation and that would have been more advantageous, is to apply FP-Growth on all the training data with all the number of stars and then filter by the consequent, where the consequent is the number of stars and the antecedent is the restaurants. Therefore, for every number of stars, we augment the graph according to the restaurants in the antecedent of the rule. This would guarantee an association between the restaurants and the number of stars and not a simple frequency of appearance. However, we leave this approach for future work.
 
## rGCN architecture
The training on the heterogeneous graph is done is using rGCN [3]. rGCN is a variation of GCN[2] that is adapted for heterogeneous and knowledge graphs. The most important difference between GCN and rGCN is that the latter introduces a weight matrix for every type of edge. Therefore, this model learns to separate between the different types of entities and relationships in the graph.

To classify the edges (number of stars per review), in the last layer we sample all the edges that are marked for training. Then, we multiply the embedding of the two nodes of the edge element-wise (Hadamard product) in order to generate an embedding for the edge. A linear transformation is then applied to the edge embeddings by multiplying them with a weight matrix and then a softmax is applied row-wise in order to generate a probability for the five types of edges. Figure 3 details the different components of the architecture.

<p align="center">
  <img width="700" height="350" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/architecture.png">
</p>
 <p align="center"><em>Figure 3 - rGCN archtiecture for edge labeling.</em></p>

## Training and results

For all our experiments we use python with the libraries: pytroch, NetworkX, and DGL (Deep Graph Library)[4] on a machine with NVIDIA GPU GeForce GTX 1050 (12 GB).  
We build the graph from the Yelp research dataset [5]. We sample 1000 users that reviewed restaurants in the Montreal area. We split the data for semi-supervised training into two batches training and testing. We take all the reviews prior to 2017 as training and all the reviews from 2017 onwards as testing. Table 1 lists the statistics of the used dataset.

<p align="center">
  <img width="40%" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/dataset%20statistics.PNG">
</p>
<p align="center"><em>Table 1 - Dataset statistics.</em></p>

Table 2 details the distribution of the classes (star numbers) in the training and test batches. We remarque that the classes are not balanced. The data augmentation that we will perform will help mitigate this problem. In order to augment the data, we test 3 minimum supports for FP-Growth. The number of edges added per minimum support is detailed in table 3.

<p align="center">
  <img width="30%" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/classes%20distribution.PNG">
</p>
<p align="center"><em>Table 2 - Classes distribution.</em></p>

<p align="center">
  <img width="50%" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/data%20augementation%20results.PNG">
</p>
<p align="center"><em>Table 3 - Data augmentation results.</em></p>

For both the GCN and rGCN models, we use two layers with a hidden layer size of 16 and ReLU activation functions. We train for 300 epochs with a patience of 30. We use the Cross-Entropy loss and the Adam optimizer with a learning rate of 0.001. We evaluate the performance of the models with RMSE (Root Mean Squared Error), which calculates the difference between the predicted and ground truth stars.  

Table 4 details the results of our experiments. The models rGCN outperform the vanilla GCN, highlighting the importance of the inclusion of the edge type in the learning process. Furthermore, the rGCN model with data augmentation having minimum support of 0.01 outperforms all the other models. This preliminary result highlights the advantages of data augmentation when performed using mined association rules.

<p align="center">
  <img width="50%" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/prediction%20examples.png">
</p>
<p align="center"><em>Table 4 - Preliminary results.</em></p>

Table 5 demonstrates some of the samples predicted by rGCN-Aug(minsup=0.01). The color-coding is as follows, Green: exact match to the stars in the review, Light brown: minor error, and Red: Major error. We find that the model tends to overestimate the results. In the example highlighted in red, there is a big difference between the score given by the model and the actual score given by the user, however, we do find that the score 5 is close to the actual general score of the restaurant on  Yelp. Also, a noticeable result for the user Emmy on the restaurant Ucan (highlighted in bold), we find that the model did a good job at low scoring the restaurant, which in practice would prevent the model from recommending a restaurant that would be disliked by the user, overall improving the user experience on the platform.

<p align="center">
  <img width="50%" src="https://github.com/MarounHaddad/Restaurant-recommendation-with-augmented-relational-graph-neural-networks/blob/main/images/prediction%20examples.png">
</p>
<p align="center"><em>Table 5 - Prediction samples by rGCN-Aug(minsup=0.01).</em></p>

## Background information
This work was presented as partial requirement for the course "INF7710 - Théorie et applications de la fouille d’associations" at UQAM (Université du Quebec à Montréal).  
Maroun Haddad (April 2020).


## References
[1] Han, J., Pei, J. and Yin, Y. (2000). Mining frequent patterns without candidate generation. In proceedings of the 2000 ACM SIGMOD International Conference on Management of Data.  
[2] Kipf, T. N. and Welling, M. (2017). Semi-supervised classification with graph convolu-tional networks. In proceedings of the 5th International Conference on Learning Representations,ICLR.  
[3] Schlichtkrull, M. S., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I. and Welling, M.(2018). Modeling relational data with graph convolutional networks. In proceedings of the 15th International Conference.  
[4] Wang, M., Yu, L., Zheng, D., Gan, Q., Gai, Y., Ye, Z., Li, M., Zhou, J., Huang, Q., Ma,C., Huang, Z., Guo, Q., Zhang, H., Lin, H., Zhao, J., Li, J., Smola, A. J. and Zhang, Z.(2019). Deep graph library : Towards efficient and scalable deep learning on graphs.  
[5] Yelp dataset. Acquired from https://www.yelp.com/dataset  
