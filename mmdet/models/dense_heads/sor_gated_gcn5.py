import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.cnn import build_norm_layer


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


class GNNLayer(nn.Module):
    def __init__(self, in_channels, num_classes, norm_cfg=dict(type='LN')):
        super(GNNLayer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        start, end = self.create_e_matrix(self.num_classes)
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
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U1.weight.data.normal_(0, scale)
        self.V1.weight.data.normal_(0, scale)
        self.A1.weight.data.normal_(0, scale)
        self.B1.weight.data.normal_(0, scale)
        self.E1.weight.data.normal_(0, scale)

        self.bn_init(self.bnv1)
        self.bn_init(self.bne1)

    def create_e_matrix(self, n):
        end = torch.zeros((n * n, n))
        for i in range(n):
            end[i * n:(i + 1) * n, i] = 1
        start = torch.zeros(n, n)
        for i in range(n):
            start[i, i] = 1
        start = start.repeat(n, 1)
        return start, end

    def bn_init(self, bn):
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()

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
        edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = self.act(res + self.bnv1(x))
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
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='SyncBN')):
        super(GNNEncoder, self).__init__()
        self.num_classes = num_classes
        self.layer1 = GNNLayer(in_dim, self.num_classes)
        self.layer2 = GNNLayer(in_dim, self.num_classes)

        self.global_layer = GFEM(in_dim, num_classes, norm_cfg=norm_cfg)
        self.edge_extractor = GEM(in_dim)

    def forward(self, x):
        res = x
        global_feature = self.global_layer(x)
        f_e = self.edge_extractor(x, global_feature)
        f_v, f_e = self.layer1(x, f_e)
        f_v, f_e = self.layer2(f_v, f_e)
        f_v = f_v + res
        return f_v


'''
norm_cfg = dict(type='GN', num_groups=32)
'''
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

        # import cv2
        # hf_mask = results['lf_mask'][0]
        # for i in range(hf_mask.shape[0]):
        #     # cv2.imwrite("/opt/data/private/mmdetection/work_dirs/test/{}.png".format(i), hf_mask[i].cpu().detach().numpy()*255)
        #     recon_mask = hf_mask[i].permute(1, 2, 0).cpu().detach().numpy()
        #     recon_mask = recon_mask[..., ::-1]
        #     print(np.unique(recon_mask * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]))
        #     cv2.imwrite("/opt/data/private/mmdetection/work_dirs/test/{}.png".format(i),
        #                 recon_mask * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53])
        # raise
        return results