import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import build_norm_layer


#Used in stage 1 (ANFL)
def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim = -1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix, A), norm_degs_matrix)
    return norm_A


#Used in stage 2 (MEFL)
def create_e_matrix(n):
    end = torch.zeros((n*n,n))
    for i in range(n):
        end[i * n:(i + 1) * n, i] = 1
    start = torch.zeros(n, n)
    for i in range(n):
        start[i, i] = 1
    start = start.repeat(n, 1)
    return start, end


def bn_init(bn):
    bn.weight.data.fill_(1)
    bn.bias.data.zero_()


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.01, norm_cfg=dict(type='LN')):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = build_norm_layer(norm_cfg, out_features)[1]
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x)
        x = self.relu(self.bn(x))
        return x


class FFM(nn.Module):
    def __init__(self, d_in, d_out, expansion_rate=4, drop=0.2, activation=F.relu, layer_norm_eps=1e-5):
        super(FFM, self).__init__()
        self.linear1 = nn.Linear(d_in, d_in*expansion_rate)
        self.dropout = nn.Dropout(drop)
        self.linear2 = nn.Linear(d_in*expansion_rate, d_out)
        self.norm = nn.LayerNorm(d_out, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(drop)

        self.activation = activation

        if d_in != d_out:
            self.res = nn.Linear(d_in, d_out)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = self.norm(res + self._ff_block(x))
        return x

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout1(x)


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots', norm_cfg=dict(type='LN')):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels, self.in_channels)
        self.V = nn.Linear(self.in_channels, self.in_channels)
        # self.bnv = nn.BatchNorm1d(num_classes)
        self.bnv = build_norm_layer(norm_cfg, self.in_channels)[1]

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'l1':
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()

        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN_EDGE(nn.Module):
    def __init__(self, in_channels, num_nodes, norm_cfg=dict(type='LN')):
        super(GNN_EDGE, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_nodes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne1 = build_norm_layer(norm_cfg, dim_out)[1]
        # self.bnv1 = nn.InstanceNorm1d(dim_out)
        # self.bne1 = nn.InstanceNorm1d(dim_out)

        self.bnv2 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne2 = build_norm_layer(norm_cfg, dim_out)[1]
        # self.bnv2 = nn.InstanceNorm1d(dim_out)
        # self.bne2 = nn.InstanceNorm1d(dim_out)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_nodes, self.num_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_nodes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_nodes, self.num_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_nodes  # V x H_out
        x = self.act(res + self.bnv2(x))
        return x, edge


class GNN_EDGE_1layer(nn.Module):
    def __init__(self, in_channels, num_nodes, norm_cfg=dict(type='LN')):
        super(GNN_EDGE_1layer, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_nodes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne1 = build_norm_layer(norm_cfg, dim_out)[1]

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_nodes, self.num_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_nodes  # V x H_out
        x = self.act(res + self.bnv1(x))

        return x, edge


class GNN_EDGE_3layer(nn.Module):
    def __init__(self, in_channels, num_nodes, norm_cfg=dict(type='LN')):
        super(GNN_EDGE_3layer, self).__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_nodes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U1 = nn.Linear(dim_in, dim_out, bias=False)
        self.V1 = nn.Linear(dim_in, dim_out, bias=False)
        self.A1 = nn.Linear(dim_in, dim_out, bias=False)
        self.B1 = nn.Linear(dim_in, dim_out, bias=False)
        self.E1 = nn.Linear(dim_in, dim_out, bias=False)

        self.U2 = nn.Linear(dim_in, dim_out, bias=False)
        self.V2 = nn.Linear(dim_in, dim_out, bias=False)
        self.A2 = nn.Linear(dim_in, dim_out, bias=False)
        self.B2 = nn.Linear(dim_in, dim_out, bias=False)
        self.E2 = nn.Linear(dim_in, dim_out, bias=False)

        self.U3 = nn.Linear(dim_in, dim_out, bias=False)
        self.V3 = nn.Linear(dim_in, dim_out, bias=False)
        self.A3 = nn.Linear(dim_in, dim_out, bias=False)
        self.B3 = nn.Linear(dim_in, dim_out, bias=False)
        self.E3 = nn.Linear(dim_in, dim_out, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.bnv1 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne1 = build_norm_layer(norm_cfg, dim_out)[1]
        # self.bnv1 = nn.InstanceNorm1d(dim_out)
        # self.bne1 = nn.InstanceNorm1d(dim_out)

        self.bnv2 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne2 = build_norm_layer(norm_cfg, dim_out)[1]
        # self.bnv2 = nn.InstanceNorm1d(dim_out)
        # self.bne2 = nn.InstanceNorm1d(dim_out)

        self.bnv3 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne3 = build_norm_layer(norm_cfg, dim_out)[1]

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.U2.weight.data.normal_(0, scale)
        self.V2.weight.data.normal_(0, scale)
        self.A2.weight.data.normal_(0, scale)
        self.B2.weight.data.normal_(0, scale)
        self.E2.weight.data.normal_(0, scale)

        self.U3.weight.data.normal_(0, scale)
        self.V3.weight.data.normal_(0, scale)
        self.A3.weight.data.normal_(0, scale)
        self.B3.weight.data.normal_(0, scale)
        self.E3.weight.data.normal_(0, scale)

        bn_init(self.bnv1)
        bn_init(self.bne1)
        bn_init(self.bnv2)
        bn_init(self.bne2)
        bn_init(self.bnv3)
        bn_init(self.bne3)

    def forward(self, x, edge):
        # device
        dev = x.get_device()
        if dev >= 0:
            start = self.start.to(dev)
            end = self.end.to(dev)

        # GNN Layer 1:
        res = x
        Vix = self.A1(x)  # V x d_out
        Vjx = self.B1(x)  # V x d_out
        e = self.E1(edge)  # E x d_out
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_nodes, self.num_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_nodes  # V x H_out
        x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_nodes, self.num_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_nodes  # V x H_out
        x = self.act(res + self.bnv2(x))
        res = x

        # GNN Layer 3:
        Vix = self.A3(x)  # V x d_out
        Vjx = self.B3(x)  # V x d_out
        e = self.E3(edge)  # E x d_out
        edge = edge + self.act(self.bne3(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_nodes, self.num_nodes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V3(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U3(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_nodes  # V x H_out
        x = self.act(res + self.bnv3(x))
        return x, edge


class GEM(nn.Module):
    def __init__(self, in_channels):
        super(GEM, self).__init__()
        self.FAM = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.ARM = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)

        self.edge_proj = FFM(in_channels, in_channels)

    def forward(self, class_feature, global_feature):
        B, N, D = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D)
        feat = self.FAM(class_feature, global_feature, global_feature, need_weights=False)[0]

        feat_end = feat.repeat(1, 1, N).view(B, -1, D)
        feat_start = feat.repeat(1, N, 1).view(B, -1, D)
        feat = self.ARM(feat_start, feat_end, feat_end, need_weights=False)[0]

        edge = self.edge_proj(feat)
        return edge


'''
Domain Separation Module
GNNEncoder:
DomainDecoder:
DomainSepModule:
'''


class GFEM(nn.Module):
    def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='SyncBN')):
        super(GFEM, self).__init__()

        self.gfe1 = nn.Conv1d(in_dim, 4 * in_dim, kernel_size=num_nodes, padding=0)
        self.norm1 = build_norm_layer(norm_cfg, 4 * in_dim)[1]

        self.gfe2 = nn.Conv1d(4 * in_dim, in_dim, kernel_size=1, padding=0)
        self.norm2 = build_norm_layer(norm_cfg, in_dim)[1]

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """

        Args:
            x[Tensor]: shape[batch, num_nodes, c]

        Returns: global feature with shape [batch, 1, c]

        """
        x = x.permute(0, 2, 1)
        gf = self.act(self.norm1(self.gfe1(x)))
        gf = self.act(self.norm2(self.gfe2(gf)))
        return gf.permute(0, 2, 1).contiguous()


# class GNNEncoder(nn.Module):
#
#     def __init__(self, in_dim, num_nodes, neighbor_num=5, metric='dots', norm_cfg=dict(type='SyncBN')):
#         super(GNNEncoder, self).__init__()
#         self.in_channels = in_dim
#         self.num_nodes = num_nodes
#         self.gnn1 = GNN(self.in_channels, self.num_nodes, neighbor_num=neighbor_num, metric=metric)
#         self.gnn2 = GNN(self.in_channels, self.num_nodes, neighbor_num=neighbor_num, metric=metric)
#         self.ffm = FFM(self.in_channels, self.in_channels)
#
#     def forward(self, nodes):
#         x = nodes
#         f_v = self.gnn1(nodes)
#         f_v = self.gnn2(f_v)
#         cl = self.ffm(f_v+x)
#         return cl


# class GNNEncoder(nn.Module):
#     def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='SyncBN')):
#         super(GNNEncoder, self).__init__()
#         self.ffm = FFM(in_dim, in_dim)
#
#     def forward(self, x):
#         f_v = self.ffm(x)
#         return f_v


# class GNNEncoder(nn.Module):
#     def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='SyncBN')):
#         super(GNNEncoder, self).__init__()
#         self.transformer_block = nn.TransformerEncoderLayer(in_dim, 8, batch_first=True)
#
#     def forward(self, x):
#         f_v = self.transformer_block(x)
#         return f_v


# class GNNEncoder(nn.Module):
#     def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='SyncBN')):
#         super(GNNEncoder, self).__init__()
#         self.cnn_block = nn.Sequential(
#             nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1),
#             nn.Conv1d(in_dim, in_dim, kernel_size=1, padding=0)
#         )
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         res = x
#         f_v = self.cnn_block(x) + res
#         return f_v.permute(0, 2, 1)



class GNNEncoder(nn.Module):
    def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='SyncBN')):
        super(GNNEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.layers = GNN_EDGE(in_dim, self.num_nodes)
        # self.layers = GNN_EDGE_3layer(in_dim, self.num_nodes)
        self.global_layer = GFEM(in_dim, num_nodes, norm_cfg=norm_cfg)
        self.edge_extractor = GEM(in_dim)

        self.ffm = FFM(in_dim, in_dim)

    def forward(self, x):
        res = x
        global_feature = self.global_layer(x)
        f_e = self.edge_extractor(x, global_feature)
        f_v, f_e = self.layers(x, f_e)
        f_v = f_v + res

        f_v = self.ffm(f_v)

        return f_v


class GraphSepModule(nn.Module):
    def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='GN', num_groups=8)):
        super(GraphSepModule, self).__init__()
        self.squ = LinearBlock(in_dim, 32)
        self.enc = GNNEncoder(in_dim=32, num_nodes=num_nodes, norm_cfg=norm_cfg)

    def forward(self, x):
        x = self.squ(x)
        sep_node = self.enc(x)
        noi_node = x - sep_node
        gnn_sep_results = dict(
            sep_node=sep_node,
            noi_node=noi_node
        )
        return gnn_sep_results


'''
norm_cfg = dict(type='GN', num_groups=32)
'''
class DomainSepModule(nn.Module):
    def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='GN', num_groups=32)):
        super(DomainSepModule, self).__init__()

        self.in_dim = in_dim

        self.hf_rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)
        self.lf_rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)

        self.shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)

        self.lf_shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)
        self.hf_shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)
        # self.task_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)
        self.hf_private_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)
        self.lf_private_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)

        self.shared_gnn_dec = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)

        self.rank_embed = LinearBlock(in_dim*2, in_dim, drop=0.)
        self.restore_hf = LinearBlock(in_dim*2, in_dim, drop=0.)
        self.restore_lf = LinearBlock(in_dim*2, in_dim, drop=0.)

        self.gnn_sep = GraphSepModule(in_dim=in_dim, num_nodes=num_nodes)

        self.rank_head_hf = nn.Linear(in_dim, 1)
        self.rank_head_lf = nn.Linear(in_dim, 1)
        self.rank_head_final = nn.Linear(32, 1)

        self.cls_head_hf = nn.Linear(in_dim, 2)
        self.cls_head_lf = nn.Linear(in_dim, 2)
        self.cls_head_final = nn.Linear(32, 2)

        self._init_weights()

    def _init_weights(self):
        self.rank_head_lf.weight.data.normal_(0, 0.02)
        self.rank_head_hf.weight.data.normal_(0, 0.02)
        self.rank_head_final.weight.data.normal_(0, 0.02)

    def forward(self, structure_feature, texture_feature):

        hf_feature = self.hf_rank_gnn_enc(structure_feature)
        lf_feature = self.lf_rank_gnn_enc(texture_feature)

        hf_shared_feature = self.shared_gnn_enc(hf_feature)
        lf_shared_feature = self.shared_gnn_enc(lf_feature)

        hf_private_feature = self.hf_private_gnn_enc(hf_feature)
        lf_private_feature = self.lf_private_gnn_enc(lf_feature)

        restored_hf_feature = self.restore_hf(torch.cat([hf_private_feature, hf_shared_feature], dim=-1))
        restored_lf_feature = self.restore_lf(torch.cat([lf_private_feature, lf_shared_feature], dim=-1))
        restored_hf_feature = self.shared_gnn_dec(restored_hf_feature)  # TODO: concat
        restored_lf_feature = self.shared_gnn_dec(restored_lf_feature)

        lf_shared_feature = self.lf_shared_gnn_enc(lf_shared_feature)
        hf_shared_feature = self.hf_shared_gnn_enc(hf_shared_feature)

        rank_pred_lf = self.rank_head_hf(lf_shared_feature).squeeze(-1)
        rank_pred_hf = self.rank_head_lf(hf_shared_feature).squeeze(-1)

        cls_pred_lf = self.cls_head_lf(lf_shared_feature)
        cls_pred_hf = self.cls_head_hf(hf_shared_feature)

        shared_feature = self.rank_embed(torch.cat([lf_shared_feature, hf_shared_feature], dim=-1))
        # shared_feature = self.task_gnn_enc(shared_feature)
        gnn_sep_results = self.gnn_sep(shared_feature)
        pred_feature = gnn_sep_results['sep_node']
        rank_pred = self.rank_head_final(pred_feature).squeeze(-1)
        cls_pred = self.cls_head_final(pred_feature)

        results = dict(
            hf_feature=hf_feature,
            lf_feature=lf_feature,
            hf_shared_feature=hf_shared_feature,
            lf_shared_feature=lf_shared_feature,
            hf_private_feature=hf_private_feature,
            lf_private_feature=lf_private_feature,
            restored_hf_feature=restored_hf_feature,
            restored_lf_feature=restored_lf_feature,
            rank_pred_lf=rank_pred_lf,
            rank_pred_hf=rank_pred_hf,
            pred_feature=pred_feature,
            rank_pred=rank_pred,
            cls_pred_lf=cls_pred_lf,
            cls_pred_hf=cls_pred_hf,
            cls_pred=cls_pred,
            gnn_sep_results=gnn_sep_results)

        return results


class DomainSepModule2(nn.Module):
    def __init__(self, in_dim, num_nodes, norm_cfg=dict(type='GN', num_groups=32)):
        super(DomainSepModule2, self).__init__()

        self.in_dim = in_dim

        self.shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_nodes=num_nodes, norm_cfg=norm_cfg)
        self.gnn_sep = GraphSepModule(in_dim=in_dim, num_nodes=num_nodes)
        self.rank_embed = LinearBlock(in_dim * 2, in_dim, drop=0.)

        self.rank_head_hf = nn.Linear(in_dim, 1)
        self.rank_head_lf = nn.Linear(in_dim, 1)
        self.rank_head_final = nn.Linear(32, 1)

        self.cls_head_hf = nn.Linear(in_dim, 2)
        self.cls_head_lf = nn.Linear(in_dim, 2)
        self.cls_head_final = nn.Linear(32, 2)

        self._init_weights()

    def _init_weights(self):
        self.rank_head_lf.weight.data.normal_(0, 0.02)
        self.rank_head_hf.weight.data.normal_(0, 0.02)
        self.rank_head_final.weight.data.normal_(0, 0.02)

    def forward(self, structure_feature, texture_feature):

        sep_node_hf = self.shared_gnn_enc(structure_feature)
        sep_node_lf = self.shared_gnn_enc(texture_feature)
        noi_node_hf = structure_feature - sep_node_hf
        noi_node_lf = texture_feature - sep_node_lf

        rank_pred_lf = self.rank_head_hf(sep_node_hf).squeeze(-1)
        rank_pred_hf = self.rank_head_lf(sep_node_lf).squeeze(-1)

        cls_pred_lf = self.cls_head_lf(sep_node_hf)
        cls_pred_hf = self.cls_head_hf(sep_node_lf)

        shared_feature = self.rank_embed(torch.cat([sep_node_hf, sep_node_lf], dim=-1))
        gnn_sep_results = self.gnn_sep(shared_feature)
        pred_feature = gnn_sep_results['sep_node']
        rank_pred = self.rank_head_final(pred_feature).squeeze(-1)
        cls_pred = self.cls_head_final(pred_feature)

        results = dict(
            hf_shared_feature=sep_node_hf,
            lf_shared_feature=sep_node_lf,
            hf_private_feature=noi_node_hf,
            lf_private_feature=noi_node_lf,
            rank_pred_lf=rank_pred_lf,
            rank_pred_hf=rank_pred_hf,
            pred_feature=pred_feature,
            rank_pred=rank_pred,
            cls_pred_lf=cls_pred_lf,
            cls_pred_hf=cls_pred_hf,
            cls_pred=cls_pred,
            gnn_sep_results=gnn_sep_results)

        return results