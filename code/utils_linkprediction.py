"""
This file contains the functions that build the graph
"""
import networkx as nx
import pandas as pd
import sklearn.metrics as metrics
import torch

import pretreatment.apriori as ap
import pretreatment.utils as ut


def build_graph(graph_type="simple"):
    # load data
    business_QC = pd.read_pickle(ut.datapath + 'business_rest')
    review_QC = pd.read_pickle(ut.datapath + 'review_rest')
    user_QC = pd.read_pickle(ut.datapath + 'user_rest')

    # add zone filed
    business_QC["zone"] = ""
    business_QC["zone"] = business_QC["postal_code"].str.split(" ", n=1, expand=True)

    user_QC = user_QC.head(1000)
    business_QC = business_QC

    # user_QC = user_QC
    # business_QC = business_QC

    review_QC = review_QC[review_QC.user_id.isin(user_QC.user_id) & review_QC.business_id.isin(business_QC.business_id)]

    print("users", len(user_QC))
    print("business", len(business_QC))

    user_QC = user_QC[user_QC.user_id.isin(review_QC.user_id)]
    business_QC = business_QC[business_QC.business_id.isin(review_QC.business_id)]

    print("users", len(user_QC))
    print("business", len(business_QC))

    # print(business_QC[business_QC.business_id=='qdG4tYLHKJ33Rut7BQM-Zg'].stars)
    # print(business_QC[business_QC.business_id=='-1xuC540Nycht_iWFeJ-dw'].stars)

    # the training views are those before the year 2016
    review_train = review_QC[review_QC.year < 2017]

    # the test reviews are those after 2016
    review_test = review_QC[review_QC.year >= 2017]

    print(review_train.groupby(["stars"])['stars'].count())
    print(review_test.groupby(["stars"])['stars'].count())

    # # balance edges
    # review_train1 = review_train[review_train.stars==1].head(1000)
    # review_train2 = review_train[review_train.stars==2].head(1000)
    # review_train3 = review_train[review_train.stars==3].head(1000)
    # review_train4 = review_train[review_train.stars==4].head(1000)
    # review_train5 = review_train[review_train.stars==5].head(1000)
    #
    # review_train =  review_train1.append([review_train2,review_train3,review_train4,review_train5])
    #
    # review_test1 = review_test[review_test.stars==1].head(400)
    # review_test2 = review_test[review_test.stars==2].head(400)
    # review_test3 = review_test[review_test.stars==3].head(400)
    # review_test4 = review_test[review_test.stars==4].head(400)
    # review_test5 = review_test[review_test.stars==5].head(400)

    # review_test =  review_test1.append([review_test2,review_test3,review_test4,review_test5])


    # leave only the test reviews that are between the users and businesses in the training dataset
    review_test = review_test[review_test.business_id.isin(review_train.business_id)]

    print("review_QC", len(review_QC))
    print("review_train", len(review_train))
    print("review_test", len(review_test))

    # build graph
    graph = nx.Graph()
    graph.add_nodes_from('a_' + user_QC.user_id)
    graph.add_nodes_from('b_' + business_QC.business_id)
    print("sanity check:")
    print("-----------------")
    print("nodes before adding edges", len(graph.nodes))

    # nodes lookup
    nodes_lookup = {}
    node_index = 0
    for node in sorted(graph.nodes):
        nodes_lookup[node] = node_index
        node_index += 1

    print(review_train.groupby(["stars"])['stars'].count())
    print(review_test.groupby(["stars"])['stars'].count())

    # test edges
    test_edges = []
    test_users = []
    test_businesses = []
    for index, row in review_test.iterrows():
        test_edges.append(str(nodes_lookup['a_' + row.user_id]) + "_" + str(nodes_lookup['b_' + row.business_id]))
        test_users.append(nodes_lookup['a_' + row.user_id])
        test_businesses.append(nodes_lookup['b_' + row.business_id])

    for index, row in review_test.iterrows():
        # add test edeges
        graph.add_edge('a_' + row.user_id, 'b_' + row.business_id, weight=row.stars, rel_type=0,
                       review_edge=0, norm=[1.])

    for index, row in review_train[review_train.stars == 1].iterrows():
        # add edge of type review
        graph.add_edge('a_' + row.user_id, 'b_' + row.business_id, weight=row.stars, rel_type=0,
                       review_edge=1, norm=[1.])

    for index, row in review_train[review_train.stars == 2].iterrows():
        # add edge of type review
        graph.add_edge('a_' + row.user_id, 'b_' + row.business_id, weight=row.stars, rel_type=1,
                       review_edge=1, norm=[1.])

    for index, row in review_train[review_train.stars == 3].iterrows():
        # add edge of type review
        graph.add_edge('a_' + row.user_id, 'b_' + row.business_id, weight=row.stars, rel_type=2,
                       review_edge=1, norm=[1.])

    for index, row in review_train[review_train.stars == 4].iterrows():
        # add edge of type review
        graph.add_edge('a_' + row.user_id, 'b_' + row.business_id, weight=row.stars, rel_type=3,
                       review_edge=1, norm=[1.])

    for index, row in review_train[review_train.stars == 5].iterrows():
        # add edge of type review
        graph.add_edge('a_' + row.user_id, 'b_' + row.business_id, weight=row.stars, rel_type=4,
                       review_edge=1, norm=[1.])

    review_edges_number = len(graph.edges)
    print("review edges", review_edges_number)

    business_attributes = build_attributes(business_QC)

    # augment data
    for star_filter in range(1, 6):
        print("augmenting:", star_filter)
        frequent_restaurants = ap.apriori_augment(review_train, star_filter)
        print("frequent:", len(frequent_restaurants))
        apriori_edges = 0
        for index, user in user_QC.iterrows():
            for restaurants in frequent_restaurants:
                if len(review_train[review_train.business_id == restaurants[0]]) > 0:
                    if not graph.has_edge('a_' + user.user_id, 'b_' + restaurants[1]):
                        graph.add_edge('a_' + user.user_id, 'b_' + restaurants[1], weight=star_filter,
                                       rel_type=star_filter - 1, review_edge=1, norm=[1.])
                        apriori_edges += 1
                elif len(review_train[review_train.business_id == restaurants[1]]) > 0:
                    if not graph.has_edge('a_' + user.user_id, 'b_' + restaurants[0]):
                        graph.add_edge('a_' + user.user_id, 'b_' + restaurants[0], weight=star_filter,
                                       rel_type=star_filter - 1, review_edge=1, norm=[1.])
                        apriori_edges += 1
        print(star_filter, "star", "apriori edges:", apriori_edges)
        print(".............")


    # # add friend edges
    # friend_edges = 0
    # for index, first in user_QC.iterrows():
    #     friends = first.friends.split(',')
    #     user_friends = user_QC[user_QC.user_id.isin(friends)]
    #     for index2, second in user_friends.iterrows():
    #         if first.user_id == second.user_id:
    #             continue
    #         graph.add_edge('a_' + first.user_id, 'a_' + second.user_id, weight=1, rel_type=6, review_edge=0, norm=[1.])
    #         friend_edges += 1

    # print("friend edges:", friend_edges)

    print("nodes after adding edges", len(graph.nodes))
    print("-----------------")

    print("graph built!")

    return graph, nodes_lookup, review_test, review_train, review_edges_number, test_edges, test_users, test_businesses, business_attributes,business_QC


def evaluate_stars(gae, embedding, nodes_lookup, review_test):
    # test positive
    predicted = []
    ground_truth = []
    for index, row in review_test.iterrows():
        node1 = torch.tensor(embedding[nodes_lookup['a_' + row.user_id]])
        node2 = torch.tensor(embedding[nodes_lookup['b_' + row.business_id]])
        predicted.append(gae.dec.forward(0, torch.mul(node1, node2), "test"))
        ground_truth.append(row.stars - 1)

    # print results
    accuracy_score = metrics.accuracy_score(ground_truth, predicted)
    precision, recall, f_measure, support = metrics.precision_recall_fscore_support(ground_truth, predicted)
    rmse_score = metrics.mean_squared_error(ground_truth, predicted)
    return accuracy_score, precision, recall, f_measure, support, rmse_score


def build_attributes(business_QC):
    """
    This function builds the attribute matrix for training the GCN
    :param business_QC: the list of businesses
    :return: attribute matrix
    """
    business_attributes = business_QC[["business_id", "stars"]]
    business_categories = business_QC["categories"].str.split('\s*,\s*', expand=True).stack()
    business_categories = pd.crosstab(business_categories.index.get_level_values(0), business_categories.values).iloc[:,
                          1:]
    business_categories.drop(columns=["Restaurants"])
    business_categories.drop(columns=["Food"])
    business_attributes = pd.concat([business_attributes, business_categories.reindex(business_attributes.index)],
                                    axis=1)

    business_QC["RestaurantsTakeOut"] = business_QC.apply(ut.get_attribute, args=("RestaurantsTakeOut", 0,), axis=1)
    business_QC["RestaurantsGoodForGroups"] = business_QC.apply(ut.get_attribute,
                                                                args=("RestaurantsGoodForGroups", 0,),
                                                                axis=1)
    business_QC["RestaurantsReservations"] = business_QC.apply(ut.get_attribute,
                                                               args=("RestaurantsReservations", 0,),
                                                               axis=1)
    business_QC["RestaurantsPriceRange2"] = business_QC.apply(ut.get_attribute, args=("RestaurantsPriceRange2", 0,),
                                                              axis=1)
    business_QC["OutdoorSeating"] = business_QC.apply(ut.get_attribute, args=("OutdoorSeating", 0,), axis=1)
    business_QC["GoodForKids"] = business_QC.apply(ut.get_attribute, args=("GoodForKids", 0,), axis=1)
    business_QC["RestaurantsDelivery"] = business_QC.apply(ut.get_attribute, args=("RestaurantsDelivery", 0,),
                                                           axis=1)
    business_attributes = pd.merge(business_attributes, business_QC[
        ['business_id', 'RestaurantsTakeOut', 'RestaurantsGoodForGroups', 'RestaurantsReservations',
         'RestaurantsPriceRange2', 'OutdoorSeating', 'GoodForKids', 'RestaurantsDelivery']], on='business_id',
                                   how='left')

    business_attributes = business_attributes.drop('business_id', axis=1)
    return business_attributes
