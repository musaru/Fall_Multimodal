import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import copy
#from model.layers import *
#from utils import import_class
#from Utils import Graph
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
def count_params(model):
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and 'fc' not in name)
def activation_factory(name, inplace=False):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'hardswish':
        return nn.Hardswish()
    elif name == 'metaacon':
        return MetaAconC()
    elif name == 'acon':
        return AconC()
    elif name == 'linear' or name is None:
        return nn.Identity()
    else:
        raise ValueError('Not supported activation:', name)
        
class Randomized_DropBlock_Ske(nn.Module):
    def __init__(self, block_size=7):
        super(Randomized_DropBlock_Ske, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob, A, num_point):  # n,c,t,v
        self.keep_prob = keep_prob
        self.num_point = num_point
        if not self.training or self.keep_prob == 1:
            return input
        n, c, t, v = input.size()
        #print(input.shape) 
        input_abs = torch.mean(torch.mean(
            torch.abs(input), dim=2), dim=1).detach()
        input_abs = input_abs / torch.sum(input_abs) * input_abs.numel()
        if self.num_point == 25:  # Kinect V2 
            gamma = (1. - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:  # Kinect V1
            gamma = (1. - self.keep_prob) / (1 + 1.9)
        else:
            gamma = (1. - self.keep_prob) / (1 + 1.92)
            warnings.warn('undefined skeleton graph')
        M_seed = torch.bernoulli(torch.clamp(
            input_abs * gamma, max=1.0)).to(device=input.device, dtype=input.dtype)
        #print(M_seed.shape)
        #print(A.shape)
        M = torch.matmul(M_seed, A)
        #M = torch.einsum('nv,cvw->nv', (M_seed, A)).contiguous()
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        #print(M.shape)
        mask = (1 - M).view(n, 1, 1, self.num_point)
        #print(M.shape)
        return input * mask * mask.numel() / mask.sum()
    
    
class Randomized_DropBlockT_1d(nn.Module):
    def __init__(self, block_size=7):
        super(Randomized_DropBlockT_1d, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size

    def forward(self, input, keep_prob):
        self.keep_prob = keep_prob
        if not self.training or self.keep_prob == 1:
            return input
        n,c,t,v = input.size()

        input_abs = torch.mean(torch.mean(torch.abs(input),dim=3),dim=1).detach()
        input_abs = (input_abs/torch.sum(input_abs)*input_abs.numel()).view(n,1,t)
        gamma = (1. - self.keep_prob) / self.block_size
        input1 = input.permute(0,1,3,2).contiguous().view(n,c*v,t)
        M = torch.bernoulli(torch.clamp(input_abs * gamma, max=1.0)).repeat(1,c*v,1)
        Msum = F.max_pool1d(M, kernel_size=[self.block_size], stride=1, padding=self.block_size // 2)
        idx = torch.randperm(Msum.shape[2])
        RMsum = Msum[:,:,idx].view(Msum.size()) ## shuffles MSum to drop random frames instead of dropping a block of frames
        mask = (1 - RMsum).to(device=input.device, dtype=input.dtype)
        #print(mask.shape)
        return (input1 * mask * mask.numel() /mask.sum()).view(n,c,v,t).permute(0,1,3,2)
    
    
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, act_type, keep_prob, block_size, 
                 num_point, residual=True, **kwargs):
        super(SpatialGraphConv, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.A = nn.Parameter(A, requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
            
        self.act = activation_factory(act_type)   
        self.bn = nn.BatchNorm2d(out_channel)
        
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )
        self.dropS = Randomized_DropBlock_Ske()
        self.dropT = Randomized_DropBlockT_1d(block_size=block_size)
         
    def forward(self, x):
        res = self.residual(x)
        #print("res.shape",res.shape)
        x = self.gcn(x)
        #print("x.shape",x.shape)
        n, kc, t, v = x.size()
        #print(" n, kc, t, v ", n, kc, t, v )
        #x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v) ######
        #print((self.A*self.edge).shape)#torch.Size([3, 14, 14])
        #print(x.shape)#torch.Size([32, 128, 29, 14])
        
        #print(" n, c, t, v ",n, kc, t, v )
        #print("c,v,w",c,v,w)
        #print(" n, c, t, w ",n, c, t, w )
        
        x = torch.einsum('nctv,cvw->nctw', (x, self.A * self.edge)).contiguous()#####
        #print(self.A * self.edge)
        #x = self.dropS(self.bn(x), self.keep_prob, self.A * self.edge, self.num_point) + self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point)
        x = self.dropT(self.dropS(self.bn(x), self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob) + self.dropT(self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob)
        
        return self.act(x)
    
class SepTemporal_Block(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act_type, edge, A, num_point, keep_prob, block_size, expand_ratio, stride=1, residual=True, **kwargs):
        super(SepTemporal_Block, self).__init__()
        self.keep_prob = keep_prob
        self.num_point = num_point
        padding = (temporal_window_size - 1) // 2
        self.act = activation_factory(act_type)

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = lambda x:0
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )
        self.A = nn.Parameter(A, requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1
        self.dropS = Randomized_DropBlock_Ske()
        self.dropT = Randomized_DropBlockT_1d(block_size=block_size)
        
    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        #x = self.dropT(x, self.keep_prob) + self.dropT(res, self.keep_prob)
        x = self.dropT(self.dropS(x, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob) + self.dropT(self.dropS(res, self.keep_prob, self.A * self.edge, self.num_point), self.keep_prob)
        return self.act(x)
        
class adjGraph():
    """ The Graph to model the skeletons extracted by the openpose
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='ntu-rgb+d',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        # elif layout=='customer settings'
        #     pass
        elif layout == 'coco_cut':
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                             (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)]
            self.edge = self_link + neighbor_link
            self.center = 13
        else:
            raise ValueError("Do Not Exist This Layout.")

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
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
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
        else:
            raise ValueError("Do Not Exist This Strategy")


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

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        #self.bn = nn.BatchNorm1d(dim* 25)#dim=2
        self.bn = nn.BatchNorm1d(dim* 14)#dim=2

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x
    
class embed(nn.Module):
    def __init__(self, dim, dim1, att_type, norm = False, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(                        
                norm_data(dim),
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        #self.attention =  Attention_Layer(dim1,  att_type=att_type)

    def forward(self, x):
        #print("Jont embedding x.shape", x.shape)
        x = self.cnn(x)
        #print(x.shape)
        return x#self.attention(x)


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
####
class DepthWiseSeparableConv_3x1_1x1(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
       
        self.seq = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=(3,1),padding=(1,0), groups=in_features),
            nn.BatchNorm2d(in_features),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, out_features, kernel_size=1),
            nn.BatchNorm2d(out_features),
            #nn.ReLU(),
        )
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.relu(self.seq(x))
        #x = self.seq(x)
        return x


####
class DepthWiseSeparableConv_1x1_1x1(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
       
        self.seq = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=(1,1),padding=0, groups=in_features),
            nn.BatchNorm2d(in_features),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, out_features, kernel_size=1),
            nn.BatchNorm2d(out_features),
            #nn.ReLU(),
        )
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.relu(self.seq(x))
        #x = self.seq(x)
        return x

####
class Sep_TCN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        middle_features = int((out_features - in_features)/2)+in_features
        self.sep31 = DepthWiseSeparableConv_3x1_1x1(in_features,middle_features)
        self.sep11 = DepthWiseSeparableConv_1x1_1x1(middle_features,out_features)
        self.shortcut = nn.Conv2d(in_features, out_features, kernel_size=1)
        
    def forward(self,x):
        res = self.shortcut(x)
        x = self.sep31(x)
        x = self.sep11(x)
        x = x + res
        return x
####
class Classification_Module(nn.Module):
    def __init__(self, in_features: int, numclass: int):
        super().__init__()
       
        self.seq = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, numclass)
        )
    def forward(self, x):
        x = self.seq(x)
        return x
####
class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 max_frame,
                 graph, 
                 bias,
                 edge,
                 block_size,
                 embed_dim=32,
                 n_stage=2,
                 act_type='relu'
                ):
        super(Model, self).__init__()
        
        self.num_classes =  num_class
        temporal_window_size = 3
        max_graph_distance = 2
        keep_prob = 0.9
        
        Graph = graph
        A_binary = torch.Tensor(Graph.A)
        
        
        self.joint_embed_pos = embed(3, embed_dim, att_type='stja', norm=False, bias=bias)
        self.joint_embed_mos = embed(2, embed_dim, att_type='stja', norm=False, bias=bias)

        
        stream_pos = []
        stream_mot = []
        for i in range(n_stage):
            stream_pos += [
                SpatialGraphConv(embed_dim, embed_dim*2, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True),
            ]
            stream_mot += [
                SpatialGraphConv(embed_dim, embed_dim*2, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True),
            ]
            embed_dim = embed_dim*2

        stream_pos += [Sep_TCN(embed_dim,embed_dim*2)]
        stream_mot += [Sep_TCN(embed_dim,embed_dim*2)]
            
        self.stream_pos = nn.Sequential(*stream_pos)
        self.stream_mot = nn.Sequential(*stream_mot)
        
        # self.Sep_TCN1 = Sep_TCN(embed_dim,embed_dim*2)
        # self.Sep_TCN2 = Sep_TCN(embed_dim,embed_dim*2)
        
        self.fc = Classification_Module((embed_dim * 4)+3, num_class)

    
    def forward(self, x):   
        pts = x
        mot = x[:,:2,:-1] - x[:,:2,1:]
        
        N, C, T, V = pts.size() #Only_pos
        N, C, T, V = mot.size()#mot
        
        res_pos = pos_p = pts.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        pos_m = mot.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        
        pos_p = self.joint_embed_pos(pos_p)   
        pos_m = self.joint_embed_mos(pos_m)
        
    
        dy = pos_p.permute(0,1,3,2).contiguous() # N, C, T, V
        dy2 = pos_m.permute(0,1,3,2).contiguous() # N, C, T, V
      
        #########################
        out = self.stream_pos(dy)
        out2 = self.stream_mot(dy2)
        ###Sep TCN
        # out = self.Sep_TCN1(out)
        # out2 = self.Sep_TCN2(out2)
        
        out_channels = out.size(1)
        out2_channels = out2.size(1)
        
        out = out.reshape(N, out_channels, -1)   
        out2 = out2.reshape(N, out2_channels, -1)
        #print("out.reshape(N, out_channels, -1)",out.shape)#torch.Size([1, 256, 1875])
        out = out.mean(2)
        out2 = out2.mean(2)
        
        res_pos_channels = res_pos.size(1)
        res_pos = res_pos.reshape(N, res_pos_channels, -1)
        res_pos = res_pos.mean(2)
        
        concat = torch.cat([out, out2, res_pos], dim=-1)
        ###Attention
        
        out = self.fc(concat)
        ###Attention
        return out



class Ablation(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 max_frame,
                 graph, 
                 bias,
                 edge,
                 block_size,
                 embed_dim=32,
                 n_stage=2,
                 act_type='relu'
                ):
        super(Ablation, self).__init__()
        
        self.num_classes =  num_class
        temporal_window_size = 3
        max_graph_distance = 2
        keep_prob = 0.9
        
        Graph = graph
        A_binary = torch.Tensor(Graph.A)
        
        
        self.joint_embed_pos = embed(3, embed_dim, att_type='stja', norm=False, bias=bias)
        self.joint_embed_mos = embed(2, embed_dim, att_type='stja', norm=False, bias=bias)

        
        stream_pos = []
        stream_mot = []
        for i in range(n_stage):
            stream_pos += [
                SpatialGraphConv(embed_dim, embed_dim*2, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True),
            ]
            stream_mot += [
                SpatialGraphConv(embed_dim, embed_dim*2, max_graph_distance, bias, edge, A_binary, act_type, keep_prob, block_size, num_point, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=1, residual=True),
                SepTemporal_Block(embed_dim*2, temporal_window_size+2, bias, act_type, edge, A_binary, num_point, keep_prob, block_size, expand_ratio=0, stride=2, residual=True),
            ]
            embed_dim *= 2


        # Sep_TCN 
        # stream_pos += [Sep_TCN(embed_dim,embed_dim*2)]
        # stream_mot += [Sep_TCN(embed_dim,embed_dim*2)]
        # embed_dim *= 2
        
        self.stream_pos = nn.Sequential(*stream_pos)
        self.stream_mot = nn.Sequential(*stream_mot)
    
        # self.fc = Classification_Module(embed_dim + 3, num_class)
        self.fc = Classification_Module(embed_dim*2+3, num_class)

    
    def forward(self, x):   
        pts = x
        mot = x[:,:2,:-1] - x[:,:2,1:]
        
        N, C, T, V = pts.size() #Only_pos
        _pos_p = pts.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        pos_p = self.joint_embed_pos(_pos_p)   
        d_pts = pos_p.permute(0,1,3,2).contiguous() # N, C, T, V
        out_pts = self.stream_pos(d_pts)
        out_pts_channels = out_pts.size(1)
        out_pts = out_pts.reshape(N, out_pts_channels, -1)   
        out_pts = out_pts.mean(2)

        
        N, C, T, V = mot.size() #mot
        pos_m = mot.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        pos_m = self.joint_embed_mos(pos_m)
        d_mot = pos_m.permute(0,1,3,2).contiguous() # N, C, T, V
        out_mot = self.stream_mot(d_mot)
        out_mot_channels = out_mot.size(1)
        out_mot = out_mot.reshape(N, out_mot_channels, -1)
        out_mot = out_mot.mean(2)
        
        res_pos = _pos_p
        res_pos_channels = res_pos.size(1)
        res_pos = res_pos.reshape(N, res_pos_channels, -1)
        res_pos = res_pos.mean(2)
        
        
        concat = torch.cat([
            out_pts, 
            out_mot, 
            res_pos
        ], dim=-1)
        
        ###Attention
        out = self.fc(concat)
        ###Attention
        return out