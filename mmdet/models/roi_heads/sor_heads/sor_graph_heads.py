# from warnings import warn
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.utils import _pair
# from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
# from mmdet.models.builder import HEADS, build_loss
# from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms

# from torch_geometric.nn import GCNConv, GATConv, ResGatedGraphConv
# from .gcnconv2d import GCNConv2d
# from .gatconv2d import GATConv2d
# from .ggcnconv2d import ResGatedGraphConv2d
# from torch_geometric.data import Data
# from torch_geometric.data.batch import Batch


# @HEADS.register_module()
# class GraphSorHead(BaseModule):
#     def __init__(self,
#                  graph_layer=None,
#                  num_layers=1,
#                  is_graph2d=False,
#                  in_channels=256,
#                  roi_feat_size=7,
#                  loss_rank=None,
#                  init_cfg=None):

#         if loss_rank is None:
#             loss_rank = dict(
#                 type='RelationLoss', loss_weight=2.0)
#         assert init_cfg is None, 'To prevent abnormal initialization ' \
#                                  'behavior, init_cfg is not allowed to be set'
#         super(GraphSorHead, self).__init__(init_cfg)

#         graph_conv = nn.ModuleList()
#         graph_norm = nn.ModuleList()
#         if graph_layer is None and not is_graph2d:
#             for i in range(num_layers):
#                 graph_conv.append(GCNConv(in_channels, in_channels))
#                 graph_norm.append(nn.BatchNorm1d(in_channels))
#         if graph_layer is not None and not is_graph2d:
#             for i in range(num_layers):
#                 graph_conv.append(self._get_graph_layer(graph_layer, in_channels))
#                 graph_norm.append(nn.BatchNorm1d(in_channels))
#         if graph_layer is not None and is_graph2d:
#             for i in range(num_layers):
#                 graph_conv.append(self._get_graph_layer(graph_layer, in_channels))
#                 graph_norm.append(nn.BatchNorm2d(in_channels))

#         self.graph_conv = graph_conv
#         self.graph_norm = graph_norm
#         self.act = nn.ReLU(True)
#         self.num_layers = num_layers
#         # WARN: roi_feat_size is reserved and not used
#         self.roi_feat_size = _pair(roi_feat_size)
#         self.in_channels = in_channels
#         self.fp16_enabled = False
#         self.loss_rank = build_loss(loss_rank)
#         self.is_graph2d = is_graph2d

#         self.liner = nn.Linear(in_channels * roi_feat_size ** 2, in_channels)
#         self.norm = nn.BatchNorm1d(in_channels)
#         self.predictor = nn.Linear(in_channels, 1)

#     @staticmethod
#     def _get_graph_layer(graph_layer, in_channels):
#         if graph_layer in ['GCNConv', 'ResGatedGraphConv', 'GCNConv2d']:
#             return globals()[graph_layer](in_channels, in_channels)
#         elif graph_layer in ['GATConv', 'GATConv2d']:
#             return globals()[graph_layer](
#                 in_channels, in_channels,
#                 heads=8, concat=False, alpha_sum=False)
#         elif graph_layer == 'ResGatedGraphConv2d':
#             return globals()[graph_layer](
#                 in_channels, in_channels, avg_pooled=False)
#         else:
#             raise NotImplementedError('Please check the name of the graph conv method.')

#     def get_targets(self, sampling_results, gt_ranks, concat=True):
#         pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
#         pos_is_gt_list = [res.pos_is_gt for res in sampling_results]
#         gt_rank_list = [rank for rank in gt_ranks]
#         gt_ranks, = multi_apply(
#             self.get_target_single,
#             pos_assigned_gt_inds_list,
#             pos_is_gt_list,
#             gt_rank_list
#         )
#         gt_ranks = [x for x in gt_ranks if len(x) > 0]
#         if concat:
#             gt_ranks = torch.cat(gt_ranks, 0)
#         return gt_ranks

#     def get_target_single(self, pos_assigned_gt_inds, pos_is_gt, gt_rank):
#         gt_inds = [pos_assigned_gt_inds[i] for i, p in enumerate(pos_is_gt) if p == 1]
#         gt_rank = [gt_rank[i] for i in gt_inds]
#         gt_rank = torch.tensor(gt_rank)
#         return gt_rank,

#     @force_fp32(apply_to=('rank_pred', ))
#     def loss(self, rank_pred, rank_labels):
#         loss = dict()
#         loss_rank = self.loss_rank(rank_labels, rank_pred)
#         loss['loss_rank'] = loss_rank
#         return loss

#     def forward(self, x, rois):
#         if not self.is_graph2d:
#             x = x.view(x.size(0), -1)
#             x = self.act(self.norm(self.liner(x)))

#         batch_inds = torch.tensor([int(rois[i][0].item()) for i in range(len(rois))])
#         batch_feats = []
#         for i in torch.unique(batch_inds):
#             feats = [x[n] for n, b in enumerate(batch_inds) if b == i]
#             feats = torch.stack(feats, dim=0)
#             graph_edge_index = self._get_edge_index(feats.size(0)).to(feats.device)
#             graph_feats = Data(x=feats, edge_index=graph_edge_index)
#             batch_feats.append(graph_feats)

#         graph_data = Batch.from_data_list(batch_feats)
#         graph_feats = graph_data.x
#         for i in range(self.num_layers):
#             graph_feats = self.graph_conv[i](graph_feats, graph_data.edge_index)
#             graph_feats = self.graph_norm[i](graph_feats)
#             graph_feats = self.act(graph_feats)

#         if self.is_graph2d:
#             graph_feats = graph_feats.view(graph_feats.size(0), -1)
#             graph_feats = self.act(self.norm(self.liner(graph_feats)))

#         score = self.predictor(graph_feats).squeeze(1)

#         scores = []
#         for i in torch.unique(batch_inds):
#             score_ = [score[n] for n, b in enumerate(batch_inds) if b == i]
#             score_ = torch.stack(score_, dim=0)
#             scores.append(score_)
#         return scores

#     def _get_edge_index(self, num_classes):
#         node_id = [i for i in range(num_classes)]

#         u = []
#         v = []
#         for i in node_id:
#             u += node_id
#             v += [i] * num_classes

#         return torch.tensor([u, v])








