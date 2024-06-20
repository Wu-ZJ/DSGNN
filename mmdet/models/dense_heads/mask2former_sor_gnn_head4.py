# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)

from mmcv.ops import point_sample
from mmcv.runner import force_fp32, ModuleList

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.utils import preprocess_panoptic_gt, get_uncertain_point_coords_with_randomness
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmcv.cnn import build_norm_layer
from torch.autograd import Variable
import numpy as np

@HEADS.register_module()
class Mask2FormerGNNHead4(AnchorFreeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 loss_mse=None,
                 loss_rank=None,
                 loss_domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]

        # build structure and texture transformer decoder
        self.structure_transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.texture_transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.structure_transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)

        self.structure_query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.structure_query_feat = nn.Embedding(self.num_queries, feat_channels)
        self.texture_query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.texture_query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        # self.structure_cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        # self.structure_rank_embed = nn.Linear(feat_channels, 1)
        # self.structure_mask_embed = nn.Sequential(
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, out_channels))
        self.structure_mask_embed = StructureDomainSepModule(feat_channels, self.num_queries)

        # self.texture_cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        # self.texture_rank_embed = nn.Linear(feat_channels, 1)
        # self.texture_mask_embed = nn.Sequential(
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        #     nn.Linear(feat_channels, out_channels))
        self.texture_mask_embed = TextureDomainSepModule(feat_channels, self.num_queries)
        self.rank_head = FusionModule(feat_channels, num_queries)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            # structure assigner & sampler
            self.structure_assigner = build_assigner(self.train_cfg.structure_assigner)
            self.structure_sampler = build_sampler(self.train_cfg.structure_sampler, context=self)
            # texture assigner & sampler
            self.texture_assigner = build_assigner(self.train_cfg.texture_assigner)
            self.texture_sampler = build_sampler(self.train_cfg.texture_sampler, context=self)
            # rank assigner & sampler
            self.rank_assigner = build_assigner(self.train_cfg.rank_assigner)
            # self.rank_sampler = build_sampler(self.train_cfg.rank_sampler, context=self)

            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.loss_mse = build_loss(loss_mse)
        self.loss_rank = build_loss(loss_rank)
        self.loss_domain = build_loss(loss_domain)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.structure_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        for p in self.texture_transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        return labels, masks

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    rank_labels_list, gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list,
         ranks_list, rank_weights_list,
         mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_target_single,
                                                     cls_scores_list, mask_preds_list,
                                                     gt_labels_list, rank_labels_list,
                                                     gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                ranks_list, rank_weights_list,
                mask_targets_list, mask_weights_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred,
                           gt_labels, rank_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        # assign and sample
        assign_result = self.structure_assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.structure_sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # rank target
        ranks = rank_labels.new_full((self.num_queries, ),
                                     0,
                                     dtype=torch.float32)
        ranks[pos_inds] = rank_labels[sampling_result.pos_assigned_gt_inds]
        rank_weights = rank_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, ranks, rank_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def get_texture_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                            rank_labels_list, gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list,
         ranks_list, rank_weights_list,
         mask_targets_list, mask_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_texture_target_single,
                                                     cls_scores_list, mask_preds_list,
                                                     gt_labels_list, rank_labels_list,
                                                     gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                ranks_list, rank_weights_list,
                mask_targets_list, mask_weights_list,
                num_total_pos, num_total_neg)

    def _get_texture_target_single(self, cls_score, mask_pred,
                                   gt_labels, rank_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, h, w).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, 3, num_points)
        mask_points_pred = point_sample(
            mask_pred, point_coords.repeat(num_queries, 1, 1))
        # shape (num_gts, 3, num_points)
        gt_points_masks = point_sample(
            gt_masks.float(), point_coords.repeat(num_gts, 1, 1))

        # assign and sample
        assign_result = self.texture_assigner.assign(cls_score, mask_points_pred,
                                                     gt_labels, gt_points_masks,
                                                     img_metas)
        sampling_result = self.texture_sampler.sample(assign_result, mask_pred,
                                                      gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)

        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # rank target
        ranks = rank_labels.new_full((self.num_queries, ),
                                     0,
                                     dtype=torch.float32)
        ranks[pos_inds] = rank_labels[sampling_result.pos_assigned_gt_inds]
        rank_weights = rank_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, ranks, rank_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def get_domain_targets(self, cls_scores_list, structure_mask_pred_list, texture_mask_pred_list,
                           gt_labels_list, rank_labels_list,
                           structure_gt_masks_list, texture_gt_masks_list, img_metas):

        (labels_list, label_weights_list,
         ranks_list, rank_weights_list,
         mask_targets_list, mask_weights_list,
         text_mask_targets_list, text_mask_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_domain_target_single,
                                                     cls_scores_list, structure_mask_pred_list,
                                                     texture_mask_pred_list, gt_labels_list,
                                                     rank_labels_list,
                                                     structure_gt_masks_list, texture_gt_masks_list,
                                                     img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                ranks_list, rank_weights_list,
                mask_targets_list, mask_weights_list,
                text_mask_targets_list, text_mask_weights_list,
                num_total_pos, num_total_neg)

    def _get_domain_target_single(self, cls_score, structure_mask_pred, texture_mask_pred,
                                  gt_labels, rank_labels, structure_gt_masks, texture_gt_masks, img_metas):

        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, 3, num_points)
        text_mask_points_pred = point_sample(
            texture_mask_pred, point_coords.repeat(num_queries, 1, 1))
        # shape (num_gts, 3, num_points)
        text_gt_points_masks = point_sample(
            texture_gt_masks.float(), point_coords.repeat(num_gts, 1, 1))

        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            structure_mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            structure_gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)

        # assign and sample
        assign_result = self.rank_assigner.assign(cls_score, mask_points_pred, text_mask_points_pred,
                                                  gt_labels, gt_points_masks, text_gt_points_masks, img_metas)

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # rank target
        ranks = rank_labels.new_full((self.num_queries, ),
                                     0,
                                     dtype=torch.float32)
        ranks[pos_inds] = rank_labels[pos_assigned_gt_inds]
        rank_weights = rank_labels.new_ones((self.num_queries, ))

        # structure mask target
        mask_targets = structure_gt_masks[pos_assigned_gt_inds]
        mask_weights = structure_mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        # texture mask target
        text_mask_targets = texture_gt_masks[pos_assigned_gt_inds]
        text_mask_weights = texture_mask_pred.new_zeros((self.num_queries, ))
        text_mask_weights[pos_inds] = 1.0

        return (labels, label_weights, ranks, rank_weights,
                mask_targets, mask_weights,
                text_mask_targets, text_mask_weights,
                pos_inds, neg_inds)

    def structure_loss_single(self, structure_results,
                              gt_labels_list, rank_labels_list,
                              gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        mask_preds = structure_results['mask_pred']
        cls_scores = structure_results['cls_pred']
        rank_scores = structure_results['rank_pred']
        task_mask_preds = structure_results['task_mask_pred']
        noise_mask_preds = structure_results['noise_mask_pred']

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]

        (labels_list, label_weights_list,
         ranks_list, rank_weights_list,
         mask_targets_list, mask_weights_list,
         num_total_pos, num_total_neg) = \
            self.get_targets(cls_scores_list, mask_preds_list,
                             gt_labels_list, rank_labels_list, gt_masks_list, img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)

        # shape (batch_size, num_queries)
        ranks = torch.stack(ranks_list, dim=0)
        # shape (batch_size, num_queries)
        rank_weights = torch.stack(rank_weights_list, dim=0)

        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        # relation loss
        # shape (batch_size, num_queries)
        loss_rank = self.loss_rank(ranks, rank_scores)

        loss_domain = self.loss_domain(structure_results)

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        task_mask_preds = task_mask_preds[mask_weights > 0]
        noise_mask_preds = noise_mask_preds[mask_weights > 0]

        noise_targets = torch.ones_like(mask_targets, dtype=torch.float32)
        noise_targets[mask_targets == 1] = 0.
        noise_targets = F.interpolate(noise_targets.unsqueeze(1),
                                      scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(1)
        noise_targets = torch.sigmoid(noise_targets * mask_preds)

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            loss_dice_task = task_mask_preds.sum()
            loss_mask_task = task_mask_preds.sum()
            loss_mse_noise = noise_mask_preds.sum()

            return (loss_cls, loss_rank, loss_domain, loss_mask, loss_dice,
                    loss_dice_task, loss_mask_task, loss_mse_noise)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            points_coords_task = get_uncertain_point_coords_with_randomness(
                task_mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            points_coords_noise = get_uncertain_point_coords_with_randomness(
                noise_mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)

            noise_point_targets = point_sample(
                noise_targets.unsqueeze(1).float(), points_coords_noise).squeeze(1)

        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        task_mask_point_preds = point_sample(
            task_mask_preds.unsqueeze(1), points_coords_task).squeeze(1)

        noise_mask_point_preds = point_sample(
            noise_mask_preds.unsqueeze(1), points_coords_noise).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        loss_dice_task = self.loss_dice(
            task_mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        task_mask_point_preds = task_mask_point_preds.reshape(-1)
        noise_mask_point_preds = noise_mask_point_preds.reshape(-1)

        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        noise_point_targets = noise_point_targets.reshape(-1)

        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        loss_mask_task = self.loss_mask(
            task_mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        loss_mse_noise = self.loss_mse(
            noise_mask_point_preds,
            noise_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return (loss_cls, loss_rank, loss_domain, loss_mask, loss_dice,
                loss_dice_task, loss_mask_task, loss_mse_noise)

    def get_texture_labels(self, img_list, gt_masks_list):
        """

        Args:
            img_list(list[Tensor]): each with shape (3, h, w).
            gt_masks_list(list[Tensor]): each with shape (num_gts, h, w).

        Returns(list[Tensor]): each with shape (num_gts, 3, h, w).

        """
        texture_labels_list = []
        for i in range(len(img_list)):
            img = img_list[i]
            masks = gt_masks_list[i]
            text_objs = []
            for obj in masks:
                text_obj = obj * img
                text_objs.append(text_obj)
            if len(text_objs) > 0:
                text_objs = torch.stack(text_objs, dim=0)
            else:
                text_objs = masks.unsqueeze(1).repeat(1, 3, 1, 1)
            texture_labels_list.append(text_objs)
        return texture_labels_list

    def texture_loss_single(self, texture_results,
                            gt_labels_list, rank_labels_list,
                            gt_masks_list, img_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """

        mask_preds = texture_results['mask_pred']
        cls_scores = texture_results['cls_pred']
        rank_scores = texture_results['rank_pred']
        task_mask_preds = texture_results['task_mask_pred']
        noise_mask_preds = texture_results['noise_mask_pred']

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        texture_labels_list = self.get_texture_labels(img_list, gt_masks_list)

        (labels_list, label_weights_list,
         ranks_list, rank_weights_list,
         mask_targets_list, mask_weights_list,
         num_total_pos, num_total_neg) = \
            self.get_texture_targets(cls_scores_list, mask_preds_list,
                                     gt_labels_list, rank_labels_list, texture_labels_list, img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)

        # shape (batch_size, num_queries)
        ranks = torch.stack(ranks_list, dim=0)
        # shape (batch_size, num_queries)
        rank_weights = torch.stack(rank_weights_list, dim=0)

        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        # relation loss
        # shape (batch_size, num_queries)
        loss_rank = self.loss_rank(ranks, rank_scores)

        loss_domain = self.loss_domain(texture_results)

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, 3, h, w) -> (num_total_gts, 3, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_preds = mask_preds.view(-1, mask_preds.size(-2), mask_preds.size(-1))

        task_mask_preds = task_mask_preds[mask_weights > 0]
        task_mask_preds = task_mask_preds.view(-1, task_mask_preds.size(-2), task_mask_preds.size(-1))

        noise_mask_preds = noise_mask_preds[mask_weights > 0]
        noise_mask_preds = noise_mask_preds.view(-1, noise_mask_preds.size(-2), noise_mask_preds.size(-1))

        noise_targets = torch.ones_like(mask_targets, dtype=torch.float32)
        noise_targets[mask_targets > 0] = 0.
        noise_targets = F.interpolate(noise_targets,
                                      scale_factor=0.25, mode='bilinear', align_corners=False)
        noise_targets = noise_targets.view(-1, noise_targets.size(-2), noise_targets.size(-1)) * mask_preds

        if mask_targets.shape[0] == 0:
            # zero match
            loss_mse = mask_preds.sum()
            loss_mse_task = task_mask_preds.sum()
            loss_mse_noise = noise_mask_preds.sum()
            return (loss_cls, loss_rank, loss_domain,
                    loss_mse, loss_mse_task, loss_mse_noise)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            points_coords_task = get_uncertain_point_coords_with_randomness(
                task_mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            points_coords_noise = get_uncertain_point_coords_with_randomness(
                noise_mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_targets = mask_targets.view(-1, mask_targets.size(-2), mask_targets.size(-1))
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)

            noise_targets = noise_targets.view(-1, noise_targets.size(-2), noise_targets.size(-1))
            noise_point_targets = point_sample(
                noise_targets.unsqueeze(1).float(), points_coords_noise).squeeze(1)

        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        task_mask_point_preds = point_sample(
            task_mask_preds.unsqueeze(1), points_coords_task).squeeze(1)

        noise_mask_point_preds = point_sample(
            noise_mask_preds.unsqueeze(1), points_coords_noise).squeeze(1)

        # mse loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        task_mask_point_preds = task_mask_point_preds.reshape(-1)
        noise_mask_point_preds = noise_mask_point_preds.reshape(-1)

        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        noise_point_targets = noise_point_targets.reshape(-1)

        loss_mse = self.loss_mse(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        loss_mse_task = self.loss_mse(
            task_mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        loss_mse_noise = self.loss_mse(
            noise_mask_point_preds,
            noise_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return (loss_cls, loss_rank, loss_domain,
                loss_mse, loss_mse_task, loss_mse_noise)

    def node_loss(self, rank_results, gt_labels_list, rank_labels_list, gt_masks_list, img_list, img_metas): #TODO

        structure_mask_pred = rank_results['structure_mask_pred']
        texture_mask_pred = rank_results['texture_mask_pred']
        cls_scores = rank_results['class_pred']
        rank_scores = rank_results['rank_pred']

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        structure_mask_preds_list = [structure_mask_pred[i] for i in range(num_imgs)]
        texture_mask_preds_list = [texture_mask_pred[i] for i in range(num_imgs)]

        texture_labels_list = self.get_texture_labels(img_list, gt_masks_list)


        (labels_list, label_weights_list,
         ranks_list, rank_weights_list,
         mask_targets_list, mask_weights_list,
         text_targets_list, text_weights_list,
         num_total_pos, num_total_neg) = \
            self.get_domain_targets(cls_scores_list, structure_mask_preds_list,
                                    texture_mask_preds_list, gt_labels_list,
                                    rank_labels_list, gt_masks_list,
                                    texture_labels_list,  img_metas)

        labels = torch.stack(labels_list, dim=0)
        label_weights = torch.stack(label_weights_list, dim=0)

        ranks = torch.stack(ranks_list, dim=0)
        rank_weights = torch.stack(rank_weights_list, dim=0)

        mask_targets = torch.cat(mask_targets_list, dim=0)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        text_targets = torch.cat(text_targets_list, dim=0)
        text_weights = torch.stack(text_weights_list, dim=0)

        cls_pred = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_pred.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_pred,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        loss_rank = self.loss_rank(ranks, rank_scores)

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        structure_mask_pred = structure_mask_pred[mask_weights > 0]
        texture_mask_pred = texture_mask_pred[text_weights > 0]
        texture_mask_pred = texture_mask_pred.view(-1, texture_mask_pred.size(-2), texture_mask_pred.size(-1))

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = structure_mask_pred.sum()
            loss_mask = structure_mask_pred.sum()
            loss_mse = texture_mask_pred.sum()
            return loss_cls, loss_rank, loss_mask, loss_dice, loss_mse

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                structure_mask_pred.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            points_coords_text = get_uncertain_point_coords_with_randomness(
                texture_mask_pred.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)

            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)

            text_targets = text_targets.view(-1, text_targets.size(-2), text_targets.size(-1))
            text_point_targets = point_sample(
                text_targets.unsqueeze(1).float(), points_coords_text).squeeze(1)

        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            structure_mask_pred.unsqueeze(1), points_coords).squeeze(1)
        text_point_preds = point_sample(
            texture_mask_pred.unsqueeze(1), points_coords_text).squeeze(1)

        text_point_preds = text_point_preds.reshape(-1)
        text_point_targets = text_point_targets.reshape(-1)
        loss_mse = self.loss_mse(
            text_point_preds,
            text_point_targets,
            avg_factor=num_total_masks * self.num_points)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_rank, loss_mask, loss_dice, loss_mse

    @force_fp32(apply_to=('structure_results_list', 'texture_results_list', 'fusion_results_list', 'rank_results'))
    def loss(self, structure_results_list, texture_results_list, fusion_results_list,
             gt_labels_list, gt_masks_list, rank_labels_list, img, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        num_dec_layers = len(structure_results_list)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_rank_labels_list = [rank_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        img_list = [img for _ in range(num_dec_layers)]

        st_loss_cls, st_loss_rank, st_loss_domain, st_loss_mask, st_loss_dice, \
        st_loss_dice_task, st_loss_mask_task, st_loss_mse_noise = multi_apply(
            self.structure_loss_single, structure_results_list,
            all_gt_labels_list, all_rank_labels_list, all_gt_masks_list, img_metas_list)

        te_loss_cls, te_loss_rank, te_loss_domain, \
        te_loss_mse, te_loss_mse_task, te_loss_mse_noise = multi_apply(
            self.texture_loss_single, texture_results_list,
            all_gt_labels_list, all_rank_labels_list, all_gt_masks_list, img_list, img_metas_list)

        fu_loss_cls, fu_loss_rank, fu_loss_mask, fu_loss_dice, fu_loss_mse = multi_apply(
            self.node_loss, fusion_results_list, all_gt_labels_list,
            all_rank_labels_list, all_gt_masks_list, img_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['losses_structure_cls'] = st_loss_cls[-1]
        loss_dict['losses_structure_mask'] = st_loss_mask[-1]
        loss_dict['losses_structure_dice'] = st_loss_dice[-1]
        loss_dict['losses_structure_rank'] = st_loss_rank[-1]
        loss_dict['losses_structure_domain'] = st_loss_domain[-1]
        loss_dict['losses_structure_dice_task'] = st_loss_dice_task[-1]
        loss_dict['losses_structure_mask_task'] = st_loss_mask_task[-1]
        loss_dict['losses_structure_mse_noise'] = st_loss_mse_noise[-1]

        loss_dict['losses_texture_cls'] = te_loss_cls[-1]
        loss_dict['losses_texture_rank'] = te_loss_rank[-1]
        loss_dict['losses_texture_mse'] = te_loss_mse[-1]
        loss_dict['losses_texture_domain'] = te_loss_domain[-1]
        loss_dict['losses_texture_mse_task'] = te_loss_mse_task[-1]
        loss_dict['losses_texture_mse_noise'] = te_loss_mse_noise[-1]

        loss_dict['losses_fusion_cls'] = fu_loss_cls[-1]
        loss_dict['losses_fusion_rank'] = fu_loss_rank[-1]
        loss_dict['losses_fusion_mask'] = fu_loss_mask[-1]
        loss_dict['losses_fusion_dice'] = fu_loss_dice[-1]
        loss_dict['losses_fusion_mse'] = fu_loss_mse[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for st_cls_i, st_rank_i, st_domain_i, st_mask_i, \
            st_dice_i, st_dice_task_i, st_mask_task_i, st_mse_noise_i, \
            te_cls_i, te_rank_i, te_domain_i, ts_mse_i, te_mse_task_i, te_mse_noise, \
            fu_cls_i, fu_rank_i, fu_mask_i, fu_dice_i, fu_mse_i in zip(
                st_loss_cls[:-1], st_loss_rank[:-1], st_loss_domain[:-1],
                st_loss_mask[:-1], st_loss_dice[:-1], st_loss_dice_task[:-1],
                st_loss_mask_task[:-1], st_loss_mse_noise[:-1], te_loss_cls[:-1],
                te_loss_rank[:-1], te_loss_domain[:-1], te_loss_mse[:-1],
                te_loss_mse_task[:-1], te_loss_mse_noise[:-1], fu_loss_cls[:-1],
                fu_loss_rank[:-1], fu_loss_mask[:-1], fu_loss_dice[:-1], fu_loss_mse[:-1]):

            loss_dict[f'd{num_dec_layer}.losses_structure_cls'] = st_cls_i
            loss_dict[f'd{num_dec_layer}.losses_structure_mask'] = st_mask_i
            loss_dict[f'd{num_dec_layer}.losses_structure_dice'] = st_dice_i
            loss_dict[f'd{num_dec_layer}.losses_structure_rank'] = st_rank_i
            loss_dict[f'd{num_dec_layer}.losses_structure_domain'] = st_domain_i
            loss_dict[f'd{num_dec_layer}.losses_structure_dice_task'] = st_dice_task_i
            loss_dict[f'd{num_dec_layer}.losses_structure_mask_task'] = st_mask_task_i
            loss_dict[f'd{num_dec_layer}.losses_structure_mse_noise'] = st_mse_noise_i

            loss_dict[f'd{num_dec_layer}.losses_texture_cls'] = te_cls_i
            loss_dict[f'd{num_dec_layer}.losses_texture_rank'] = te_rank_i
            loss_dict[f'd{num_dec_layer}.losses_texture_mse'] = ts_mse_i
            loss_dict[f'd{num_dec_layer}.losses_texture_domain'] = te_domain_i
            loss_dict[f'd{num_dec_layer}.losses_texture_mse_task'] = te_mse_task_i
            loss_dict[f'd{num_dec_layer}.losses_texture_mse_noise'] = te_mse_noise

            loss_dict[f'd{num_dec_layer}.losses_fusion_cls'] = fu_cls_i
            loss_dict[f'd{num_dec_layer}.losses_fusion_rank'] = fu_rank_i
            loss_dict[f'd{num_dec_layer}.losses_fusion_mask'] = fu_mask_i
            loss_dict[f'd{num_dec_layer}.losses_fusion_dice'] = fu_dice_i
            loss_dict[f'd{num_dec_layer}.losses_fusion_mse'] = fu_mse_i

            num_dec_layer += 1

        return loss_dict

    def forward_structure_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.structure_transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)

        structure_results = self.structure_mask_embed(decoder_out, mask_feature)
        task_mask_pred = structure_results['task_mask_pred']
        attn_mask = F.interpolate(
            task_mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return structure_results, attn_mask

    def forward_texture_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.texture_transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)

        texture_results = self.texture_mask_embed(decoder_out, mask_feature)

        return texture_results

    def forward_gnn_sep_head(self, structure_res, texture_res, mask_features):
        structure_feat = structure_res['task_feature']
        texture_feat = texture_res['task_feature']
        rank_results = self.rank_head(structure_feat, texture_feat, mask_features)
        return rank_results

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(2, 0, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        # shape (num_queries, c) -> (num_queries, batch_size, c)
        # 初始化结构和纹理query_feat
        structure_query_feat = self.structure_query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        structure_query_embed = self.structure_query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        texture_query_feat = self.texture_query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        texture_query_embed = self.texture_query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        structure_results_list = []
        texture_results_list = []
        fusion_results_list = []
        structure_results, attn_mask = self.forward_structure_head(
            structure_query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

        texture_results = self.forward_texture_head(
            texture_query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

        fusion_results = self.forward_gnn_sep_head(structure_results, structure_results, mask_features)

        structure_results_list.append(structure_results)
        texture_results_list.append(texture_results)
        fusion_results_list.append(fusion_results)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            structure_layer = self.structure_transformer_decoder.layers[i]
            texture_layer = self.texture_transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]

            structure_query_feat = structure_layer(
                query=structure_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=structure_query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            texture_query_feat = texture_layer(
                query=texture_query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=texture_query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)

            structure_results, attn_mask = self.forward_structure_head(
                structure_query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            texture_results = self.forward_texture_head(
                texture_query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            fusion_results = self.forward_gnn_sep_head(structure_results, structure_results, mask_features)

            structure_results_list.append(structure_results)
            texture_results_list.append(texture_results)
            fusion_results_list.append(fusion_results)

        return structure_results_list, texture_results_list, fusion_results_list

    def forward_train(self,
                      feats,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_ranks,  # TODO
                      gt_semantic_seg,
                      gt_bboxes_ignore=None):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        structure_results_list, texture_results_list, fusion_results_list = self(feats, img_metas)

        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)

        # loss
        losses = self.loss(structure_results_list, texture_results_list, fusion_results_list,
                           gt_labels, gt_masks, gt_ranks, img, img_metas)

        return losses

    def simple_test(self, feats, img_metas, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        structure_results_list, texture_results_list, fusion_results_list = self(feats, img_metas)

        # rank_pred = fusion_results_list[-1]['rank_pred']
        # mask_pred_results = fusion_results_list[-1]['structure_mask_pred']
        rank_pred = texture_results_list[-1]['rank_pred']
        mask_pred_results = structure_results_list[-1]['mask_pred']

        # upsample masks
        img_shape = img_metas[0]['batch_input_shape']
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        # import cv2
        # text_pred_results = texture_mask_pred_list[-1]
        # text_pred_result = F.interpolate(
        #     text_pred_results[0],
        #     size=(img_shape[0], img_shape[1]),
        #     mode='bilinear',
        #     align_corners=False)
        # for i in range(len(text_pred_result)):
        #     text = text_pred_result[i].permute(1, 2, 0).cpu().detach().numpy()
        #     cv2.imwrite("/opt/data/private/mmdetection/work_dirs/rgb_test/{}.png".format(i),
        #                  text * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53])
        # raise

        # import cv2
        # text_pred_results = structure_mask_pred_list[-1]
        # text_pred_result = F.interpolate(
        #     text_pred_results[0].unsqueeze(1),
        #     size=(img_shape[0], img_shape[1]),
        #     mode='bilinear',
        #     align_corners=False)
        # for i in range(len(text_pred_result)):
        #     text = text_pred_result[i].permute(1, 2, 0).cpu().detach().numpy()
        #     cv2.imwrite("/opt/data/private/mmdetection/work_dirs/rgb_test/{}.png".format(i),
        #                  text * 255)
        # raise

        return rank_pred, mask_pred_results


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
    def __init__(self, in_features, out_features=None, drop=0.01, norm_cfg=dict(type='SyncBN')):
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
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x


# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN_EDGE(nn.Module):
    def __init__(self, in_channels, num_classes, norm_cfg=dict(type='SyncBN')):
        super(GNN_EDGE, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GNN Matrix: E x N
        # Start Matrix Item:  define the source node of one edge
        # End Matrix Item:  define the target node of one edge
        # Algorithm details in Residual Gated Graph Convnets: arXiv preprint arXiv:1711.07553
        # or Benchmarking Graph Neural Networks: arXiv preprint arXiv:2003.00982v3

        start, end = create_e_matrix(self.num_classes)
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
        # self.bne1 = nn.BatchNorm1d(num_classes*num_classes)

        self.bnv2 = build_norm_layer(norm_cfg, dim_out)[1]
        self.bne2 = build_norm_layer(norm_cfg, dim_out)[1]
        # self.bnv2 = nn.BatchNorm1d(num_classes)
        # self.bne2 = nn.BatchNorm1d(num_classes * num_classes)

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
        temp = (torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e).permute(0, 2, 1)
        edge = edge + self.act(self.bne1(temp)).permute(0, 2, 1)
        # edge = edge + self.act(self.bne1(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V1(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U1(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = x.permute(0, 2, 1)
        x = self.act(res + self.bnv1(x).permute(0, 2, 1))
        # x = self.act(res + self.bnv1(x))
        res = x

        # GNN Layer 2:
        Vix = self.A2(x)  # V x d_out
        Vjx = self.B2(x)  # V x d_out
        e = self.E2(edge)  # E x d_out
        temp = (torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e).permute(0, 2, 1)
        edge = edge + self.act(self.bne2(temp)).permute(0, 2, 1)
        # edge = edge + self.act(self.bne2(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec', (start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b, self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)

        Ujx = self.V2(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U2(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = x.permute(0, 2, 1)
        x = self.act(res + self.bnv2(x).permute(0, 2, 1))
        # x = self.act(res + self.bnv2(x))
        return x, edge


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
    def __init__(self, in_channels, num_classes, norm_cfg=dict(type='SyncBN')):
        super(GEM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        # self.edge_proj = nn.Linear(in_channels, in_channels)
        # self.bn = build_norm_layer(norm_cfg, self.num_classes * self.num_classes)[1]
        # self.bn = nn.BatchNorm1d(self.num_classes * self.num_classes)
        self.edge_proj = LinearBlock(in_channels, in_channels, drop=0., norm_cfg=norm_cfg)

        # self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        # self.bn.weight.data.fill_(1)
        # self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        B, N, D = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D)
        feat = self.FAM(class_feature, global_feature)
        feat_end = feat.repeat(1, 1, N).view(B, -1, D)
        feat_start = feat.repeat(1, N, 1).view(B, -1, D)
        feat = self.ARM(feat_start, feat_end)
        # edge = self.bn(self.edge_proj(feat))
        edge = self.edge_proj(feat)
        return edge


'''
Domain Separation Module
GNNEncoder:
DomainDecoder:
DomainSepModule:
'''


class GFEM(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='SyncBN')):
        super(GFEM, self).__init__()
        self.gfe1 = nn.Conv1d(num_classes, 1, kernel_size=3, padding=1)
        self.norm1 = build_norm_layer(norm_cfg, in_dim)[1]

        # self.attn = CrossAttn(in_dim)

        self.gfe2 = nn.Conv1d(1, 1, kernel_size=1)
        self.norm2 = build_norm_layer(norm_cfg, in_dim)[1]

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        gf = self.gfe1(x).permute(0, 2, 1)
        gf = self.norm1(gf).permute(0, 2, 1)
        gf = self.act(gf)
        # gf = self.attn(x, x)

        gf = self.gfe2(gf).permute(0, 2, 1)
        gf = self.norm2(gf).permute(0, 2, 1)
        gf = self.act(gf)
        return gf


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='SyncBN')):
        super(GNNEncoder, self).__init__()
        self.num_classes = num_classes
        self.layers = GNN_EDGE(in_dim, self.num_classes, norm_cfg=norm_cfg)
        self.global_layer = GFEM(in_dim, num_classes, norm_cfg=norm_cfg)
        self.edge_extractor = GEM(in_dim, self.num_classes, norm_cfg=norm_cfg)

    def forward(self, x):
        res = x
        global_feature = self.global_layer(x)
        f_e = self.edge_extractor(x, global_feature)
        f_v, f_e = self.layers(x, f_e)
        f_v = f_v + res
        return f_v


'''
norm_cfg = dict(type='GN', num_groups=32)
'''
class StructureDomainSepModule(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
        super(StructureDomainSepModule, self).__init__()

        self.in_dim = in_dim

        self.rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.task_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.noise_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.gnn_dec = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.restore_embed = LinearBlock(in_dim*2, in_dim, drop=0., norm_cfg=norm_cfg)
        self.final_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.rank_head = nn.Linear(in_dim, 1)
        self.cls_head = nn.Linear(in_dim, 2)

        self._init_weights()

    def _init_weights(self):
        self.rank_head.weight.data.normal_(0, 0.02)
        self.cls_head.weight.data.normal_(0, 0.02)

    def forward(self, node_feature, mask_feature):

        node_feature = self.rank_gnn_enc(node_feature)
        mask_pred = torch.einsum('bqc,bchw->bqhw', node_feature, mask_feature)

        task_feature = self.task_gnn_enc(node_feature)
        noise_feature = self.noise_gnn_enc(node_feature)
        task_mask_pred = torch.einsum('bqc,bchw->bqhw', task_feature, mask_feature)
        noise_mask_pred = torch.einsum('bqc,bchw->bqhw', noise_feature, mask_feature)

        restored_feature = self.restore_embed(torch.cat([task_feature, noise_feature], dim=-1))
        restored_feature = self.gnn_dec(restored_feature)
        restored_mask_pred = torch.einsum('bqc,bchw->bqhw', restored_feature, mask_feature)

        pred_feature = self.final_enc(task_feature)
        rank_pred = self.rank_head(pred_feature).squeeze(-1)
        cls_pred = self.cls_head(pred_feature)

        results = dict(
            node_feature=node_feature,
            mask_pred=mask_pred,
            task_feature=task_feature,
            noise_feature=noise_feature,
            task_mask_pred=task_mask_pred,
            noise_mask_pred=torch.sigmoid(noise_mask_pred),
            restored_feature=restored_feature,
            restored_mask_pred=restored_mask_pred,
            rank_pred=rank_pred,
            cls_pred=cls_pred)
        return results


class TextureDomainSepModule(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
        super(TextureDomainSepModule, self).__init__()

        self.in_dim = in_dim

        self.rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.task_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.noise_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.gnn_dec = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.restore_embed = LinearBlock(in_dim*2, in_dim, drop=0., norm_cfg=norm_cfg)
        self.final_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.rank_head = nn.Linear(in_dim, 1)
        self.cls_head = nn.Linear(in_dim, 2)

        self._init_weights()

    def _init_weights(self):
        self.rank_head.weight.data.normal_(0, 0.02)
        self.cls_head.weight.data.normal_(0, 0.02)

    def forward(self, node_feature, mask_feature):
        B, C, H, W = mask_feature.shape
        mask_feature = mask_feature.repeat(1, 3, 1, 1).view(B, C, -1, H, W)

        node_feature = self.rank_gnn_enc(node_feature)
        mask_pred = torch.einsum('bqc,bcrhw->bqrhw', node_feature, mask_feature)

        task_feature = self.task_gnn_enc(node_feature)
        noise_feature = self.noise_gnn_enc(node_feature)
        task_mask_pred = torch.einsum('bqc,bcrhw->bqrhw', task_feature, mask_feature)
        noise_mask_pred = torch.einsum('bqc,bcrhw->bqrhw', noise_feature, mask_feature)

        restored_feature = self.restore_embed(torch.cat([task_feature, noise_feature], dim=-1))
        restored_feature = self.gnn_dec(restored_feature)
        restored_mask_pred = torch.einsum('bqc,bcrhw->bqrhw', restored_feature, mask_feature)

        pred_feature = self.final_enc(task_feature)
        rank_pred = self.rank_head(pred_feature).squeeze(-1)
        cls_pred = self.cls_head(pred_feature)

        results = dict(
            node_feature=node_feature,
            mask_pred=torch.tanh(mask_pred),
            task_feature=task_feature,
            noise_feature=noise_feature,
            task_mask_pred=torch.tanh(task_mask_pred),
            noise_mask_pred=torch.tanh(noise_mask_pred),
            restored_feature=restored_feature,
            restored_mask_pred=torch.tanh(restored_mask_pred),
            rank_pred=rank_pred,
            cls_pred=cls_pred)
        return results


class FusionModule(nn.Module):
    def __init__(self, in_dim, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
        super(FusionModule, self).__init__()
        # self.gnn_encoder = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.structure_mask_embed = nn.Sequential(
            nn.Linear(in_dim*2, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim))

        self.texture_mask_embed = nn.Sequential(
            nn.Linear(in_dim*2, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim))

        self.cls_embed = nn.Sequential(
            nn.Linear(in_dim*2, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, 2))

        self.rank_embed = nn.Sequential(
            nn.Linear(in_dim*2, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, 1))

    def forward(self, structure_task_feature, texture_task_feature, mask_feature):
        """

        Args:
            structure_task_feature: shape [batch_size, num_queries, c]
            texture_task_feature: shape [batch_size, num_queries, c]
            mask_feature: shape [batch_size, c, h, w]
        Returns: dict results

        """
        task_feature = torch.cat([structure_task_feature, texture_task_feature], dim=-1)

        structure_feature = self.structure_mask_embed(task_feature)
        texture_feature = self.texture_mask_embed(task_feature)
        structure_mask_pred = torch.einsum('bqc,bchw->bqhw', structure_feature, mask_feature)
        B, C, H, W = mask_feature.shape
        mask_feature = mask_feature.repeat(1, 3, 1, 1).view(B, C, -1, H, W)
        texture_mask_pred = torch.einsum('bqc,bcrhw->bqrhw', texture_feature, mask_feature)

        class_pred = self.cls_embed(task_feature)
        rank_pred = self.rank_embed(task_feature).squeeze(-1)

        results = dict(
            structure_mask_pred=structure_mask_pred,
            texture_mask_pred=torch.tanh(texture_mask_pred),
            class_pred=class_pred,
            rank_pred=rank_pred)

        return results