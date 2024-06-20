import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import dgl
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.001):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


class CrossAttn(nn.Module):
    """ cross attention Module"""

    def __init__(self, in_channels):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out


class GEM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GEM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)

        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm1d(self.num_classes * self.num_classes)
        self.act = nn.ReLU(inplace=True)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, node_feature, global_feature):
        # compute edge representation among class features
        B, N, D = node_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D)
        feat = self.FAM(node_feature, global_feature)
        feat_end = feat.repeat(1, 1, N).view(B, -1, D)
        feat_start = feat.repeat(1, N, 1).view(B, -1, D)
        edge = self.ARM(feat_start, feat_end)
        edge = self.act(self.bn(self.edge_proj(edge)))
        return edge


def get_act_by_str(name: str, negative_slope: float = 0):
    if name == "leaky_relu":
        res = nn.LeakyReLU(negative_slope, inplace=True)
    elif name == "tanh":
        res = nn.Tanh()
    elif name == "none":
        res = nn.Identity()
    elif name == "relu":
        res = nn.ReLU()
    else:
        res = nn.Softplus()
    return res


class GIPAWideConv(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            out_feats,
            n_heads,
            edge_drop=0.0,
            negative_slope=0.2,
            activation=None,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax"
    ):
        super(GIPAWideConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._norm = norm
        self._agg_layer_norm = batch_norm
        self._edge_agg_mode = edge_agg_mode
        self._edge_att_act = edge_att_act

        # project function
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)

        # propagation function
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False) if use_attn_dst else None
        self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False) if edge_feats > 0 else None
        # self.edge_norm = nn.BatchNorm1d(edge_feats) if edge_feats > 0 else None
        self.edge_att_actv = get_act_by_str(edge_att_act, negative_slope)
        self.edge_drop = edge_drop

        #  aggregation function
        self.offset = nn.Parameter(torch.zeros(size=(1, n_heads, out_feats)))
        self.scale = nn.Parameter(torch.ones(size=(1, n_heads, out_feats)))
        self.agg_fc = nn.Linear(out_feats * n_heads, out_feats * n_heads)

        # apply function
        self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
        self.apply_fc = nn.Linear(out_feats * n_heads, out_feats)
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)

        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

    def agg_func(self, h):
        if self._agg_layer_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self._n_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._n_heads, 1) + 1e-9
            h = (h - mean) * self.scale * torch.rsqrt(var) + self.offset
        return self.agg_fc(h.view(-1, self._out_feats * self._n_heads)).view(-1, self._n_heads, self._out_feats)

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # project function: source node
            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads,
                                                     self._out_feats)  # shape [batch*num_node, num_head, num_feat]

            # propagation function: source node
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)  # shape [batch*num_node, num_head, 1]
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            # propagation function: dst node
            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)  # shape [batch*num_node, num_head, 1]
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            # propagation function: edge
            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads,
                                                              1)  # shape [batch*num_node**2, num_head, 1]
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.edge_att_actv(e)
            # Deep Attention: more fc-layer can be added after this

            # edge drop
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
            graph.edata["a"] = torch.zeros_like(e)

            # edge softmax
            if self._edge_agg_mode == "single_softmax":
                graph.edata["a"][eids] = edge_softmax(graph, e[eids], eids=eids, norm_by='dst')
            else:
                graph.edata["a"][eids] = e[eids]

            # graph normalize
            if self._norm == "adj":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm_adjust"][eids].view(-1, 1, 1)
            if self._norm == "avg":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm"][eids].view(-1, 1, 1)

            # aggregation
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))
            agg_msg = self.agg_func(graph.dstdata["feat_src_fc"])

            # apply part
            if self.dst_fc is not None:
                feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads,
                                                         self._out_feats)  # shape [batch*num_node, num_head, num_feat]
                rst = agg_msg + feat_dst_fc  # apply = fc(concat([h_{k-1}, msg]))
            else:
                rst = agg_msg
            rst = self.leaky_relu(rst)
            rst = self.apply_fc(rst.flatten(1, -1))
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)
            return rst


class GIPADeepConv(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            n_head,
            out_feats,
            edge_drop=0.0,
            negative_slope=0.2,
            activation=None,
            use_attn_dst=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax",
            use_att_edge=True,
            use_prop_edge=False,
            edge_prop_size=20
    ):
        super(GIPADeepConv, self).__init__()
        self._norm = norm
        self._batch_norm = batch_norm
        self._edge_agg_mode = edge_agg_mode
        self._use_prop_edge = use_prop_edge
        self._edge_prop_size = edge_prop_size

        # optional fc
        self.prop_edge_fc = None
        self.attn_dst_fc = None
        self.attn_edge_fc = None
        self.attn_dst_fc_e = None
        self.attn_edge_fc_e = None

        # propagation src feature
        self.prop_src_fc = nn.Linear(node_feats, n_head, bias=False)

        # attn fc
        self.attn_src_fc = nn.Linear(node_feats, n_head, bias=False)
        if use_attn_dst:
            self.attn_dst_fc = nn.Linear(node_feats, n_head, bias=False)
        if edge_feats > 0 and use_att_edge:
            self.attn_edge_fc = nn.Linear(edge_feats, n_head, bias=False)

        # msg BN
        if batch_norm:
            self.agg_batch_norm = nn.BatchNorm1d(n_head)

        # agg function
        self.agg_fc = nn.Linear(n_head, out_feats)

        # apply function
        self.apply_dst_fc = nn.Linear(node_feats, out_feats)
        self.apply_fc = nn.Linear(out_feats, out_feats)

        if use_prop_edge and edge_prop_size > 0:
            self.prop_edge_fc = nn.Linear(edge_feats, edge_prop_size, bias=False)
            self.prop_src_fc_e = nn.Linear(node_feats, edge_prop_size)
            self.attn_src_fc_e = nn.Linear(node_feats, edge_prop_size, bias=False)
            if use_attn_dst:
                self.attn_dst_fc_e = nn.Linear(node_feats, edge_prop_size, bias=False)
            if edge_feats > 0 and use_att_edge:
                self.attn_edge_fc_e = nn.Linear(edge_feats, edge_prop_size, bias=False)
            if batch_norm:
                self.agg_batch_norm_e = nn.BatchNorm1d(edge_prop_size)
            self.agg_fc_e = nn.Linear(edge_prop_size, out_feats)

        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace=True)
        self.edge_att_actv = get_act_by_str(edge_att_act, negative_slope)
        self.activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.prop_src_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_dst_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        # nn.init.zeros_(self.attn_src_fc.bias)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_dst_fc.bias)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)
            # nn.init.zeros_(self.attn_edge_fc.bias)
        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)

        if self._use_prop_edge and self._edge_prop_size > 0:
            nn.init.xavier_normal_(self.prop_src_fc_e.weight, gain=gain)
            nn.init.xavier_normal_(self.prop_edge_fc.weight, gain=gain)
            nn.init.xavier_normal_(self.attn_src_fc_e.weight, gain=gain)
            if self.attn_dst_fc_e is not None:
                nn.init.xavier_normal_(self.attn_dst_fc_e.weight, gain=gain)
            if self.attn_edge_fc_e is not None:
                nn.init.xavier_normal_(self.attn_edge_fc_e.weight, gain=gain)
            nn.init.xavier_normal_(self.agg_fc_e.weight, gain=gain)
            nn.init.zeros_(self.agg_fc_e.bias)

    def agg_function(self, h, idx):
        out = h
        if self._batch_norm:
            out = self.agg_batch_norm(h) if idx == 0 else self.agg_batch_norm_e(h)

        return self.agg_fc(out) if idx == 0 else self.agg_fc_e(out)

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # propagation value prepare
            feat_src_fc = self.prop_src_fc(feat_src)
            graph.srcdata.update({"_feat_src_fc": feat_src_fc})

            # src node attention
            attn_src = self.attn_src_fc(feat_src)
            graph.srcdata.update({"_attn_src": attn_src})

            # dst node attention
            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst)
                graph.dstdata.update({"_attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("_attn_src", "_attn_dst", "_attn_node"))
            else:
                graph.apply_edges(fn.copy_u("_attn_src", "_attn_node"))

            e = graph.edata["_attn_node"]
            if self.attn_edge_fc is not None:
                attn_edge = self.attn_edge_fc(feat_edge)
                graph.edata.update({"_attn_edge": attn_edge})
                e += graph.edata["_attn_edge"]
            e = self.edge_att_actv(e)

            if self._edge_agg_mode == "both_softmax":
                graph.edata["_a"] = torch.sqrt(edge_softmax(graph, e, norm_by='dst').clamp(min=1e-9)
                                               * edge_softmax(graph, e, norm_by='src').clamp(min=1e-9))
            elif self._edge_agg_mode == "single_softmax":
                graph.edata["_a"] = edge_softmax(graph, e, norm_by='dst')
            else:
                graph.edata["_a"] = e

            if self._norm == "adj":
                graph.edata["_a"] = graph.edata["_a"] * graph.edata["gcn_norm_adjust"].view(-1, 1)
            if self._norm == "avg":
                graph.edata["_a"] = (graph.edata["_a"] * graph.edata["gcn_norm"].view(-1, 1)) / 2

            graph.update_all(fn.u_mul_e("_feat_src_fc", "_a", "_m"), fn.sum("_m", "_feat_src_fc"))
            msg_sum = graph.dstdata["_feat_src_fc"]
            # print(msg_sum.size())
            # aggregation function
            rst = self.agg_function(msg_sum, 0)

            if self._use_prop_edge and self._edge_prop_size > 0:
                graph.edata["_v"] = self.prop_edge_fc(feat_edge)
                feat_src_fc_e = self.prop_src_fc_e(feat_src)
                graph.srcdata.update({"_feat_src_fc_e": feat_src_fc_e})
                graph.apply_edges(fn.u_add_e("_feat_src_fc_e", "_v", "_prop_edge"))

                # src node attention
                attn_src_e = self.attn_src_fc_e(feat_src)
                graph.srcdata.update({"_attn_src_e": attn_src_e})

                # dst node attention
                if self.attn_dst_fc is not None:
                    attn_dst_e = self.attn_dst_fc_e(feat_dst)
                    graph.dstdata.update({"_attn_dst_e": attn_dst_e})
                    graph.apply_edges(fn.u_add_v("_attn_src_e", "_attn_dst_e", "_attn_node_e"))
                else:
                    graph.apply_edges(fn.copy_u("_attn_src_e", "_attn_node_e"))

                e_e = graph.edata["_attn_node_e"]
                if self.attn_edge_fc is not None:
                    attn_edge_e = self.attn_edge_fc_e(feat_edge)
                    graph.edata.update({"_attn_edge_e": attn_edge_e})
                    e_e += graph.edata["_attn_edge_e"]
                e_e = self.edge_att_actv(e_e)

                if self._edge_agg_mode == "both_softmax":
                    graph.edata["_a_e"] = torch.sqrt(edge_softmax(graph, e_e, norm_by='dst').clamp(min=1e-9)
                                                     * edge_softmax(graph, e_e, norm_by='src').clamp(min=1e-9))
                elif self._edge_agg_mode == "single_softmax":
                    graph.edata["_a_e"] = edge_softmax(graph, e_e, norm_by='dst')
                else:
                    graph.edata["_a_e"] = e_e

                if self._norm == "adj":
                    graph.edata["_a_e"] = graph.edata["_a_e"] * graph.edata["gcn_norm_adjust"].view(-1, 1)
                if self._norm == "avg":
                    graph.edata["_a_e"] = (graph.edata["_a_e"] * graph.edata["gcn_norm"].view(-1, 1)) / 2

                graph.edata["_m_e"] = graph.edata["_a_e"] * graph.edata["_prop_edge"]
                graph.update_all(fn.copy_e("_m_e", "_m_copy_e"), fn.sum("_m_copy_e", "_feat_src_fc_e"))
                msg_sum_e = graph.dstdata["_feat_src_fc_e"]
                rst_e = self.agg_function(msg_sum_e, 1)
                rst += rst_e

            # apply function
            rst += self.apply_dst_fc(feat_dst)
            rst = self.leaky_relu(rst)
            rst = self.apply_fc(rst)
            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst


class GraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_gnn = GIPADeepConv(node_feats=128,
                                     edge_feats=128,
                                     n_head=8,
                                     out_feats=128,
                                     edge_drop=0,
                                     activation=F.relu,
                                     use_attn_dst=True,
                                     norm="none",
                                     batch_norm=True,
                                     edge_att_act="none",
                                     edge_agg_mode="both_softmax",
                                     use_att_edge=True,
                                     use_prop_edge=False,
                                     edge_prop_size=128)
        self.wide_gnn = GIPAWideConv(node_feats=128,
                                     edge_feats=128,
                                     out_feats=128,
                                     n_heads=8,
                                     edge_drop=0,
                                     activation=F.relu,
                                     use_attn_dst=True,
                                     norm="none",
                                     batch_norm=True,
                                     edge_att_act="none",
                                     edge_agg_mode="single_softmax")

        self.graph = self._create_graph(5, 2).to('cuda')
        self.global_layer = LinearBlock(128, 128)
        self.edge_extractor = GEM(128, 5)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act = nn.ReLU(inplace=True)

    def forward(self, node):
        global_feature = node.mean(dim=-2)
        global_feature = self.global_layer(global_feature.unsqueeze(1))
        edge = self.edge_extractor(node, global_feature)

        B = node.shape[0]
        C = node.shape[-1]
        node = node.reshape(-1, C)
        edge = edge.reshape(-1, C)
        rst1 = self.deep_gnn(self.graph, node, edge)
        rst2 = self.wide_gnn(self.graph, node, edge)
        rst = self.act(self.bn1(rst1) + self.bn2(rst2))
        return rst.reshape(B, -1, C)

    def _create_graph(self, num_nodes, batch_size):
        node_id = [i for i in range(num_nodes)]

        u = []
        v = []
        for i in node_id:
            u += [i] * num_nodes
            v += node_id

        g = dgl.graph((torch.tensor(u), torch.tensor(v)), idtype=torch.int32, num_nodes=num_nodes)
        graphs = [g for _ in range(batch_size)]
        graphs = dgl.batch(graphs)
        return graphs


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn_layer_1 = GraphModule()
        # self.gnn_layer_2 = GraphModule()

        self.bn = nn.BatchNorm1d(128)
        self.act = nn.ReLU(inplace=True)

    def forward(self, node):
        node = self.act(self.bn(node.permute(0, 2, 1)).permute(0, 2, 1))

        x = node
        node1 = self.gnn_layer_1(node) + x
        # node2 = self.gnn_layer_2(node1) + node1
        return node1


if __name__ == '__main__':
    graph_enc = GraphEncoder().cuda()
    x = torch.randn((2, 5, 1024)).cuda()
    node = graph_enc(x)
    print(node.shape)
