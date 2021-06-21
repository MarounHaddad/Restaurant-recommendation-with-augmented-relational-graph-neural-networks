"""
Model GNN for edge classification
2 model types : GCN (using custom layers) and rGCN (using rGCN layers)
"""

# import libraries
import warnings
import random

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from linkprediction import rgcn_layer as rgcn

warnings.filterwarnings("ignore")

# model type (gcn or rgnc)
model_type = "gcn"

# standard message function
gcn_message = fn.copy_src(src='h', out='m')

# reduce function (aggregation)
gcn_reduce = fn.sum(msg='m', out='h')

# list of test edges
test_edges = ()

# list of training edges
train_edges = []

# number of nodes to update with message passing
review_train_edges_number = 0

# not used (rule that uses weights for training)
def gcn_message_custom(edges):
    return {'m': edges.src['h'], 'w': edges.data['weight'].float(), 's': edges.src['deg'], 'd': edges.dst['deg']}

# not used (rule that uses weights for training)
def gcn_reduce_custom(nodes):
    return {
        'h': torch.sum(nodes.mailbox['m'] * nodes.mailbox['w'].unsqueeze(2), dim=1)}


class EncoderLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout):
        super(EncoderLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=True)
        self.activation = activation
        self.norm = nn.BatchNorm1d(out_feats)
        self.drop = nn.Dropout(dropout)

    def sample_update_edges(self, g: dgl.graph):
        # edges = random.sample(range(0, len(g.edata["weight"])),int(len(g.edges())/5))
        # g.send_and_recv(edges)
        g.send_and_recv(g.edges())

    def forward(self, g: dgl.graph, input):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = input
        # g.update_all(gcn_message_custom, gcn_reduce_custom)
        self.sample_update_edges(g)

        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        h = self.linear(h)
        h = self.activation(h)
        h = self.norm(h)
        h = self.drop(h)
        return h


class DecoderLayer(nn.Module):
    def __init__(self, inputSize, outputSize, dropout):
        super(DecoderLayer, self).__init__()
        self.var = torch.var
        self.norm = nn.BatchNorm1d(outputSize)
        self.drop = nn.Dropout(dropout)
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, g: dgl.graph, inputs, mode):
        if mode == "train":
            embedding_positive, ground_truth_positive = sample_positive(g, inputs)
            embedding = torch.stack(embedding_positive, dim=0)
            ground_truth = torch.stack(ground_truth_positive, dim=0)

            predicted = self.linear(embedding)
            predicted = torch.softmax(predicted, dim=1)
            return predicted, ground_truth
        else:
            predicted = self.linear(inputs)
            predicted = torch.softmax(predicted, dim=0).argmax()
            return predicted


class GAE(nn.Module):
    def __init__(self, number_nodes, input_size, hidden_size, encoded_size,number_relations):
        super(GAE, self).__init__()
        if model_type == "gcn":
            self.enc1 = EncoderLayer(input_size, hidden_size, torch.relu, 0)
            self.enc2 = EncoderLayer(hidden_size, encoded_size, torch.relu, 0)

        elif model_type == "rgcn":
            self.enc1 = rgcn.RGCNLayer(number_nodes, hidden_size, number_relations, -1,
                                       activation=F.relu, is_input_layer=True)
            self.enc2 = rgcn.RGCNLayer(hidden_size, encoded_size, number_relations, -1,
                                       activation=F.relu)
        self.dec = DecoderLayer(encoded_size, 5, 0)

    def forward(self, g, inputs):
        if model_type == "gcn":
            encoded1 = self.enc1.forward(g, inputs)
            encoded2 = self.enc2.forward(g, encoded1)

            embedding = torch.cat((encoded1, encoded2),
                                  dim=1)
        else:
            self.enc1.forward(g,inputs)
            self.enc2.forward(g,inputs)
            encoded2 = g.ndata.pop('h')
            embedding = encoded2

        predicted, ground_truth = self.dec.forward(g, encoded2, "train")

        return predicted, ground_truth, encoded2


def train(model_name, graph, inputs, input_size, hidden_size, embedding_size, epochs, early_stopping, test_data,
          review_edges_number, print_progress=True):
    """
    This function trains the graph autoencoder in order to generate the embeddings of the graph (a vector per node)
    :param graph: a networkx graph for a time step
    :param inputs: the attributes to be used as input
    :param input_size: the size of the input
    :param hidden_size: the hidden layer size
    :param embedding_size: the embedding (encoder output) size
    :param epochs: the number of training epochs
    :param early_stopping: the number of epochs for early stopping
    :param print_progress: whether to print the training progress or not
    :return: the embedding of the graph (a vector for every node in the graph)
    """

    global test_edges
    global sample_positive_numbers
    global sample_negative_numbers
    global model_type
    global review_train_edges_number
    global train_edges

    model_type = model_name
    test_edges = test_data
    review_train_edges_number = review_edges_number
    # generate a dgl graph object from the networkx object
    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(graph, edge_attrs=['weight', 'rel_type', 'review_edge', 'norm'])

    dgl_graph.add_edges(dgl_graph.nodes(), dgl_graph.nodes())
    dgl_graph.readonly(True)

    dgl_graph.ndata['id'] = torch.arange(len(dgl_graph.nodes()))
    dgl_graph.ndata['deg'] = dgl_graph.out_degrees(dgl_graph.nodes()).float()

    train_edges = set(dgl_graph.filter_edges(rel_edges))

    number_relations = len(set(dgl_graph.edata['rel_type'].detach().numpy()))

    # large graph sampling message passing
    m_func = dgl.function.copy_src('h', 'm')
    dgl_graph.register_message_func(m_func)
    m_reduce_func = fn.sum('m', 'h')
    dgl_graph.register_reduce_func(m_reduce_func)

    gae = GAE(graph.number_of_nodes(), input_size, hidden_size, embedding_size,number_relations)

    # for GPU
    if model_name == "gcn":
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


    gae = gae.to(device)
    dgl_graph.to(device)
    inputs = inputs.to(device)

    # optimizer used for training
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.001)

    min_loss = 1000
    stop_index = 0
    for epoch in range(epochs):
        predicted, ground_truth, embedding = gae.forward(dgl_graph, inputs)
        embedding = embedding.to(device)
        predicted = predicted.to(device)
        ground_truth = ground_truth.to(device)
        # loss = F.binary_cross_entropy_with_logits(predicted, ground_truth)
        loss = F.cross_entropy(predicted, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < min_loss:
            if print_progress:
                print('Epoch %d | old Loss: %.4f | New Loss:  %.4f ' % (epoch, min_loss, loss.item()))

            # we only save the embedding if there is an improvement in training
            save_emb = embedding
            save_pred = predicted
            min_loss = loss
            stop_index = 0
        else:
            if print_progress:
                print('Epoch %d | No improvement | Loss: %.4f | old Loss :  %.4f ' % (epoch, loss.item(), min_loss))
            stop_index += 1

        if stop_index == early_stopping:
            if print_progress:
                print("Early Stopping!")
            break

    save_emb = save_emb.detach().cpu()
    save_emb = save_emb.numpy()
    save_pred = torch.round(save_pred).detach().cpu()
    save_pred = save_pred.numpy()

    return save_emb, save_pred, gae


def sample_positive(dgl_graph, inputs):
    """
    This function sample the training edeges and calculates the embedding
    of the edge by doing element wise multiplication of the embedding of its nodes
    :param dgl_graph: the graph
    :param inputs: the embedding
    :return: edge embedding and ground truth
    """
    # positive_edges = random.sample(train_edges,int(len(train_edges)/2))
    positive_edges = random.sample(train_edges,len(train_edges))
    # positive_edges = random.sample(train_edges,500)

    nodes = dgl_graph.find_edges(positive_edges)
    edge_index = 0
    embedding = []
    ground_truth = []
    for edge in positive_edges:
        embedding.append(torch.mul(inputs[nodes[0][edge_index]], inputs[nodes[1][edge_index]]))
        ground_truth.append(torch.tensor(dgl_graph.edata["weight"][edge] - 1).long())
        edge_index += 1

    return embedding, ground_truth


def rel_edges(edges):
    return (edges.data['review_edge'] == 1).detach().cpu()
