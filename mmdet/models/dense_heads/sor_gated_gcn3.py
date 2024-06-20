import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import build_norm_layer


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0., norm_cfg=dict(type='LN')):
        super(LinearBlock, self).__init__()
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
    def __init__(self, d_in, d_out, expansion_rate=4, drop=0., activation=F.relu, layer_norm_eps=1e-5):
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


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNNLayer(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate=0.):
        super(GNNLayer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U = nn.Linear(dim_in, dim_out, bias=False)
        self.V = nn.Linear(dim_in, dim_out, bias=False)
        self.A = nn.Linear(dim_in, dim_out, bias=False)
        self.B = nn.Linear(dim_in, dim_out, bias=False)
        self.E = nn.Linear(dim_in, dim_out, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.bnv = nn.BatchNorm1d(num_classes)
        self.bne = nn.BatchNorm1d(num_classes * num_classes)
        self.act = nn.ReLU()
        self.init_weights_linear(dim_in, 1)

    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U.weight.data.normal_(0, scale)
        self.V.weight.data.normal_(0, scale)
        self.A.weight.data.normal_(0, scale)
        self.B.weight.data.normal_(0, scale)
        self.E.weight.data.normal_(0, scale)

        self.bn_init(self.bnv)
        self.bn_init(self.bne)

    def bn_init(self, bn):
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()

    def forward(self, x, edge, start, end):

        res = x
        Vix = self.A(x)  # V x d_out
        Vjx = self.B(x)  # V x d_out
        e = self.E(edge)  # E x d_out
        # print(e.shape)
        # print(x.shape)
        # print(start.shape)
        # print(end.shape)

        edge = edge + self.act(self.bne(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = res + self.act(self.bnv(x))

        return x, edge


# GAT GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, layer_num=2):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        start, end = self.create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        graph_layers = []
        for i in range(layer_num):
            layer = GNNLayer(self.in_channels, self.num_classes)
            graph_layers += [layer]

        self.graph_layers = nn.ModuleList(graph_layers)

    def forward(self, x, edge):
        dev = x.get_device()
        if dev >= 0:
            self.start = self.start.to(dev)
            self.end = self.end.to(dev)
        for i, layer in enumerate(self.graph_layers):
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge

    def create_e_matrix(self, n):
        end = torch.zeros((n * n, n))
        for i in range(n):
            end[i * n:(i + 1) * n, i] = 1
        start = torch.zeros(n, n)
        for i in range(n):
            start[i, i] = 1
        start = start.repeat(n, 1)
        return start, end


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


class GFEM(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='SyncBN')):
        super(GFEM, self).__init__()

        self.gfe1 = nn.Conv1d(in_dim, 4 * in_dim, kernel_size=num_classes, padding=0)
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


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
        super(GNNEncoder, self).__init__()

        self.global_layer = GFEM(in_dim, num_classes, norm_cfg=norm_cfg)
        self.edge_extractor = GEM(in_dim)

        self.node_att = nn.MultiheadAttention(embed_dim=in_dim, num_heads=8, batch_first=True)

        self.gnn = GNN(in_dim, num_classes, layer_num=3)

        self.ffm_node = FFM(in_dim, in_dim)

    def forward(self, node):
        n_res = node

        global_feature = self.global_layer(node)
        edge = self.edge_extractor(node, global_feature).contiguous()

        node = self.node_att(node, node, node, need_weights=False)[0]
        n_att = n_res + node

        node, edge = self.gnn(n_att, edge)

        node = n_att + node

        node = self.ffm_node(node) + n_res

        return node


class DomainSepModule(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
        super(DomainSepModule, self).__init__()

        self.in_dim = in_dim

        self.hf_rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.lf_rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.lf_shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.hf_shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.hf_private_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.lf_private_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.shared_gnn_dec = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.rank_embed = LinearBlock(in_dim*2, in_dim, drop=0.)
        self.restore_hf = LinearBlock(in_dim*2, in_dim, drop=0.)
        self.restore_lf = LinearBlock(in_dim*2, in_dim, drop=0.)

        self.final_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.rank_head_hf = nn.Linear(in_dim, 1)
        self.rank_head_lf = nn.Linear(in_dim, 1)
        self.rank_head_final = nn.Linear(in_dim, 1)

        self.cls_head_hf = nn.Linear(in_dim, 2)
        self.cls_head_lf = nn.Linear(in_dim, 2)
        self.cls_head_final = nn.Linear(in_dim, 2)

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
        pred_feature = self.final_enc(shared_feature)
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
            shared_feature=shared_feature,
            pred_feature=pred_feature,
            rank_pred=rank_pred,
            cls_pred_lf=cls_pred_lf,
            cls_pred_hf=cls_pred_hf,
            cls_pred=cls_pred)

        return results


# class GNNEncoder(nn.Module):
#     def __init__(self, in_dims, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
#         super(GNNEncoder, self).__init__()
#
#         self.global_layer = GFEM(in_dims, num_classes, norm_cfg=norm_cfg)
#         self.edge_extractor = GEM(in_dims)
#
#         self.node_att = nn.MultiheadAttention(embed_dim=in_dims, num_heads=8, batch_first=True)
#         self.edge_att = nn.MultiheadAttention(embed_dim=in_dims, num_heads=8, batch_first=True)
#
#         self.gnn = GNN(in_dims, num_classes, layer_num=4)
#
#         self.ffm_node = FFM(in_dims, in_dims)
#         self.ffm_edge = FFM(in_dims, in_dims)
#
#     def forward(self, node, edge):
#         n_res = node
#         e_res = edge
#
#         global_feature = self.global_layer(node)
#         edge = self.edge_extractor(node, global_feature).contiguous() + edge
#
#         node = self.node_att(node, node, node, need_weights=False)[0]
#         edge = self.edge_att(edge, edge, edge, need_weights=False)[0]
#
#         n_att = n_res + node
#         e_att = e_res + edge
#
#         node, edge = self.gnn(n_att, e_att)
#
#         node = n_att + node
#         edge = e_att + edge
#
#         node = self.ffm_node(node)
#         edge = self.ffm_edge(edge)
#         return node, edge
#
#
# class GNNStem(nn.Module):
#     def __init__(self, in_dims, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
#         super(GNNStem, self).__init__()
#
#         self.global_layer = GFEM(in_dims, num_classes, norm_cfg=norm_cfg)
#         self.edge_extractor = GEM(in_dims)
#
#         self.gnn = GNN(in_dims, num_classes, layer_num=2)
#
#     def forward(self, node):
#         global_feature = self.global_layer(node)
#         edge = self.edge_extractor(node, global_feature).contiguous()
#
#         node, edge = self.gnn(node, edge)
#
#         return node, edge
#
#
# class DomainSepModule(nn.Module):
#     def __init__(self, in_dims, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
#         super(DomainSepModule, self).__init__()
#
#         self.in_dim = in_dims
#
#         self.hf_rank_gnn_enc = GNNStem(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#         self.lf_rank_gnn_enc = GNNStem(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#
#         self.shared_gnn_enc = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#
#         self.lf_shared_gnn_enc = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#         self.hf_shared_gnn_enc = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#         self.hf_private_gnn_enc = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#         self.lf_private_gnn_enc = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#
#         self.shared_gnn_dec = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#
#         self.rank_embed_node = LinearBlock(in_dims*2, in_dims, drop=0.)
#         self.rank_embed_edge = LinearBlock(in_dims*2, in_dims, drop=0.)
#         self.restore_hf_node = LinearBlock(in_dims*2, in_dims, drop=0.)
#         self.restore_lf_node = LinearBlock(in_dims*2, in_dims, drop=0.)
#         self.restore_hf_edge = LinearBlock(in_dims*2, in_dims, drop=0.)
#         self.restore_lf_edge = LinearBlock(in_dims*2, in_dims, drop=0.)
#
#         self.final_enc = GNNEncoder(in_dims=in_dims, num_classes=num_classes, norm_cfg=norm_cfg)
#
#         self.rank_head_hf = nn.Linear(in_dims, 1)
#         self.rank_head_lf = nn.Linear(in_dims, 1)
#         self.rank_head_final = nn.Linear(in_dims, 1)
#
#         self.cls_head_hf = nn.Linear(in_dims, 2)
#         self.cls_head_lf = nn.Linear(in_dims, 2)
#         self.cls_head_final = nn.Linear(in_dims, 2)
#
#         self.edge_cls_hf = nn.Linear(in_dims, 3)
#         self.edge_cls_lf = nn.Linear(in_dims, 3)
#         self.edge_cls_final = nn.Linear(in_dims, 3)
#
#         self._init_weights()
#
#     def _init_weights(self):
#         self.rank_head_lf.weight.data.normal_(0, 0.02)
#         self.rank_head_hf.weight.data.normal_(0, 0.02)
#         self.rank_head_final.weight.data.normal_(0, 0.02)
#
#     def forward(self, structure_feature, texture_feature):
#
#         hf_node, hf_edge = self.hf_rank_gnn_enc(structure_feature)
#         lf_node, lf_edge = self.lf_rank_gnn_enc(texture_feature)
#
#         hf_shared_node, hf_shared_edge = self.shared_gnn_enc(hf_node, hf_edge)
#         lf_shared_node, lf_shared_edge = self.shared_gnn_enc(lf_node, lf_edge)
#
#         hf_private_node, hf_private_edge = self.hf_private_gnn_enc(hf_node, hf_edge)
#         lf_private_node, lf_private_edge = self.lf_private_gnn_enc(lf_node, lf_edge)
#
#         restored_hf_node = self.restore_hf_node(torch.cat([hf_private_node, hf_shared_node], dim=-1))
#         restored_lf_node = self.restore_lf_node(torch.cat([lf_private_node, lf_shared_node], dim=-1))
#         restored_hf_edge = self.restore_hf_edge(torch.cat([hf_private_edge, hf_shared_edge], dim=-1))
#         restored_lf_edge = self.restore_lf_edge(torch.cat([lf_private_edge, lf_shared_edge], dim=-1))
#
#         restored_hf_node, restored_hf_edge = self.shared_gnn_dec(restored_hf_node, restored_hf_edge)
#         restored_lf_node, restored_lf_edge = self.shared_gnn_dec(restored_lf_node, restored_lf_edge)
#
#         lf_shared_node, lf_shared_edge = self.lf_shared_gnn_enc(lf_shared_node, lf_shared_edge)
#         hf_shared_node, hf_shared_edge = self.hf_shared_gnn_enc(hf_shared_node, hf_shared_edge)
#
#         shared_node = self.rank_embed_node(torch.cat([lf_shared_node, hf_shared_node], dim=-1))
#         shared_edge = self.rank_embed_edge(torch.cat([lf_shared_edge, hf_shared_edge], dim=-1))
#         pred_node, pred_edge = self.final_enc(shared_node, shared_edge)
#
#         rank_pred_lf = self.rank_head_hf(lf_shared_node).squeeze(-1)
#         rank_pred_hf = self.rank_head_lf(hf_shared_node).squeeze(-1)
#         rank_pred = self.rank_head_final(pred_node).squeeze(-1)
#
#         cls_pred_lf = self.cls_head_lf(lf_shared_node)
#         cls_pred_hf = self.cls_head_hf(hf_shared_node)
#         cls_pred = self.cls_head_final(pred_node)
#
#         edge_pred_lf = self.edge_cls_lf(lf_shared_edge)
#         edge_pred_hf = self.edge_cls_hf(hf_shared_edge)
#         edge_pred = self.edge_cls_final(pred_edge)
#
#         results = dict(
#             hf_node=hf_node,
#             hf_edge=hf_edge,
#             lf_node=lf_node,
#             lf_edge=lf_edge,
#             hf_shared_node=hf_shared_node,
#             hf_shared_edge=hf_shared_edge,
#             lf_shared_node=lf_shared_node,
#             lf_shared_edge=lf_shared_edge,
#             hf_private_node=hf_private_node,
#             hf_private_edge=hf_private_edge,
#             lf_private_node=lf_private_node,
#             lf_private_edge=lf_private_edge,
#             restored_hf_node=restored_hf_node,
#             restored_hf_edge=restored_hf_edge,
#             restored_lf_node=restored_lf_node,
#             restored_lf_edge=restored_lf_edge,
#
#             rank_pred_lf=rank_pred_lf,
#             rank_pred_hf=rank_pred_hf,
#             rank_pred=rank_pred,
#             cls_pred_lf=cls_pred_lf,
#             cls_pred_hf=cls_pred_hf,
#             cls_pred=cls_pred,
#             edge_pred_lf=edge_pred_lf,
#             edge_pred_hf=edge_pred_hf,
#             edge_pred=edge_pred
#         )
#
#         return results