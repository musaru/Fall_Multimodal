import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from TRAGCN.GRU import GRU
from TRAGCN.TA import transformer_layer
from torch.autograd import Variable
import math
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("using", device, "device")
import numpy as np

class Graph:
    """The Graph to model the skeletons extracted by the Alpha-Pose.
    Args:
        - strategy: (string) must be one of the follow candidates
            - uniform: Uniform Labeling,
            - distance: Distance Partitioning,
            - spatial: Spatial Configuration,
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        - layout: (string) must be one of the follow candidates
            - coco_cut: Is COCO format but cut 4 joints (L-R ears, L-R eyes) out.
        - max_hop: (int) the maximal distance between two connected nodes.
        - dilation: (int) controls the spacing between the kernel points.
    """
    def __init__(self,
                 layout='coco_cut',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        if layout == 'coco_cut':
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                             (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)]
            self.edge = self_link + neighbor_link
            self.center = 13
        else:
            raise ValueError('This layout is not supported!')

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            #self.A = np.swapaxes(np.swapaxes(A, 0, 1), 1, 2)
        else:
            raise ValueError("This strategy is not supported!")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, adj,num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one GRU layer in the Encoder.'
        self.adj=adj
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers

        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GRU(node_num, dim_in, dim_out, self.adj,cheb_k, embed_dim))
        # self.tcn=TemporalConvNet(dim_in,[1,1,1],3,0.2)
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GRU(node_num, dim_out, dim_out,self.adj ,cheb_k, embed_dim))
        self.trans_layer_T = transformer_layer(dim_out, dim_out, 2, 2)

    def forward(self, x, init_state, node_embeddings):

        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]

        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        current_inputs=self.trans_layer_T(current_inputs)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class TARGCN(nn.Module):
    def __init__(self, input_dim=3,num_classes=11,num_nodes=14,rnn_units=64,output_dim=64,horizon=30,num_layers=2,embed_dim=64,cheb_k=2,adj=Graph(layout='coco_cut',
                 strategy='uniform').A):
        super(TARGCN, self).__init__()
        self.num_clsses = num_classes
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        # self.adj=adj
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.adj = adj if adj != None else torch.ones((self.num_node,self.num_node))
        # self.default_graph = args.default_graph

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, self.embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_node, self.input_dim, self.hidden_dim, self.cheb_k,
                                self.embed_dim,self.adj, self.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(6, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.output_dim,self.num_clsses),
        )

    def forward(self, source ):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        output = output[:, -6:, :, :]                                   #B, 6, N, hidden
        # output = self.FC(output.permute(0,3,2,1)) .permute(0,3,2,1)
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1


        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node) # b t c n
        output = output.permute(0, 1, 3, 2)                         #B, T(12), N, C
        output = self.fc(output.permute(0,3,1,2))

        return output

if __name__=='__main__':
    model = TARGCN()
    print(model)
    inputs = torch.randn(1, 3, 30, 14, 1)#batch, channel, frame, node, 1
    print(model(inputs.permute(0,4,2,3,1).view(-1,30,14,3)).size()) #(1,1,30,14,3)
#     import argparse
#     import configparser
#     config = configparser.ConfigParser()
#     config_file = './PEMSD8_AGCRN.conf'

#     config.read(config_file)

#     args = argparse.ArgumentParser(description='arguments')

#     args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
#     args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
#     args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
#     args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
#     args.add_argument('--horizon', default=config['data']['horizon'], type=int)
#     args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
#     args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
#     args.add_argument('--cheb_k', default=config['model']['cheb_order'], type=int)
#     # args.add_argument('--embed_dim', default=2, type=int)
#     args = args.parse_args()

#     num_node = args.num_nodes
#     input_dim = args.input_dim
#     hidden_dim = args.rnn_units
#     output_dim = args.output_dim
#     horizon = args.horizon
#     num_layers = args.num_layers
#     adj = torch.ones((num_node,num_node))
#     # print(adj.shape)
#     node_embeddings = nn.Parameter(torch.randn(num_node, 2), requires_grad=True)
#     agcrn=AGCRN(args,adj)
#     # source: B, T_1, N, D
#     # target: B, T_2, N, D
#     x=torch.randn(32,12,170,1)
#     tar=torch.randn(32,12,170,1)
#     out=agcrn(x,tar)
#     print(out.shape)