"""
In this file we apply the test on the graph using rGCN (heterogeneous relations)
"""
import numpy as np
import torch

from linkprediction import gcn_stars as gae
from linkprediction import utils_linkprediction as lu
from linkprediction import visualize as vs

# build graph (heterogenous)
print("heterogenous graph")
print("-----------------")
graph, nodes_lookup, review_test, review_train, review_edges_number, test_edges, test_users, test_businesses, business_attributes, business_QC = lu.build_graph(
    "heterogenous")

# attributes
# input_users = torch.eye(n=917,m=261)
# input_business = torch.from_numpy(business_attributes.values.astype(np.float32)).float()
# input_attributes = torch.cat((input_users, input_business), 0)
# input_size = 917

input_attributes = torch.eye(len(graph.nodes))
input_size = len(graph.nodes)

# scores
accuracy_scores = []
precisions = []
recalls = []
f_measures = []
supports = []
rmse_scores = []

# train model
for experiments in range(0, 1):
    print("experiment number: " + str(experiments + 1))

    embedding, new_adj, gae = gae.train("rgcn", graph, input_attributes, input_size, 16, 16, 300, 30, test_edges,
                                        review_edges_number, True)

    accuracy_score, precision, recall, f_measure, support, rmse_score = lu.evaluate_stars(gae, embedding, nodes_lookup,
                                                                                          review_test)

    accuracy_scores.append(accuracy_score)
    precisions.append(precision)
    recalls.append(recall)
    f_measures.append(f_measure)
    supports.append(support)
    rmse_scores.append(rmse_score)

edge_embeddings = []
edge_stars = []
# evaluate model
for index, row in review_test.sort_values(by=['user_id']).iterrows():
    node1 = torch.tensor(embedding[nodes_lookup['a_' + row.user_id]])
    node2 = torch.tensor(embedding[nodes_lookup['b_' + row.business_id]]).t()
    business_name = business_QC[business_QC.business_id == row.business_id].name.iloc[0]
    business_stars = business_QC[business_QC.business_id == row.business_id].stars.iloc[0]
    print(gae.dec.forward(0, torch.mul(node1, node2), "test") + 1, row.stars, row.user_id, row.business_id,
          business_name, business_stars)
    edge_embeddings.append(torch.mul(node1, node2).detach().numpy())
    edge_stars.append(row.stars - 1)

print("RMSE:" + str(np.mean(rmse_scores).round(3)))
print("Accuracy:" + str(np.mean(accuracy_scores).round(3)))
print("F_measure:" + str(np.mean(f_measures).round(3)))
print("Precision:" + str(np.mean(precisions).round(3)))
print("Recall:" + str(np.mean(recalls).round(3)))

vs.visualize_edge(edge_embeddings, edge_stars)
