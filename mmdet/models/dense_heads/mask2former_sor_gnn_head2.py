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
# import dgl
# import dgl.function as fn
# from scipy.ndimage.morphology import distance_transform_edt


@HEADS.register_module()
class Mask2FormerGNNHead2(AnchorFreeHead):
    """Implements the MaskFormer head.

    See `Per-Pixel Classification is Not All You Need for Semantic
    Segmentation <https://arxiv.org/pdf/2107.06278>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add a layer
            to change the embed_dim of tranformer encoder in pixel decoder to
            the embed_dim of transformer decoder. Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to `FocalLoss`.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to `DiceLoss`.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Maskformer head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of Maskformer
            head.
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
                    max_salient_object=10,
                    num_transformer_feat_level=3,
                    pixel_decoder=None,
                    enforce_decoder_input_project=False,
                    transformer_decoder=None,
                    positional_encoding=None,
                    loss_cls=None,
                    loss_mask=None,
                    loss_dice=None,
                    loss_rank=None,
                    train_cfg=None,
                    test_cfg=None,
                    init_cfg=None,
                    **kwargs):
            super(AnchorFreeHead, self).__init__(init_cfg)
            self.num_things_classes = num_things_classes
            self.num_stuff_classes = num_stuff_classes
            self.num_classes = self.num_things_classes + self.num_stuff_classes
            self.num_queries = num_queries

            self.max_salient_object = max_salient_object
            self.max_ranks = 0
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
            self.transformer_decoder = build_transformer_layer_sequence(
                transformer_decoder)
            self.decoder_embed_dims = self.transformer_decoder.embed_dims

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
            self.query_embed = nn.Embedding(self.num_queries, feat_channels)
            self.query_feat = nn.Embedding(self.num_queries, feat_channels)
            # from low resolution to high resolution
            self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                            feat_channels)

            self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)

            # add node task
            # self.node_embed = nn.Sequential(
            #     nn.Linear(feat_channels, 1024), nn.ReLU(inplace=True))
            self.node_embed = LinearBlock(feat_channels, 256, drop=0., norm_cfg=dict(type='GN', num_groups=32))
            # self.rank_head = RankHead(in_channels=1024, num_classes=max_salient_object, neighbor_num=max_salient_object)
            # self.rank_head = RankHead(in_channels=256, num_classes=max_salient_object)
            self.rank_head = DomainSepModule(in_dim=256, hidden_dim=256, num_classes=max_salient_object)
            # self.rank_head = nn.Linear(256, 1)
            self.mask_embed = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, out_channels))

            self.test_cfg = test_cfg
            self.train_cfg = train_cfg
            if train_cfg:
                self.assigner = build_assigner(self.train_cfg.assigner)
                self.sor_assigner = build_assigner(self.train_cfg.sor_assigner)
                self.sampler = build_sampler(self.train_cfg.sampler, context=self)
                self.sor_sampler = build_sampler(self.train_cfg.sor_sampler, context=self)
                self.num_points = self.train_cfg.get('num_points', 12544)
                self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
                self.importance_sample_ratio = self.train_cfg.get(
                    'importance_sample_ratio', 0.75)

            self.class_weight = loss_cls.class_weight
            # self.class_weight = loss_cls.get('class_weight', None)
            self.loss_cls = build_loss(loss_cls)
            self.loss_mask = build_loss(loss_mask)
            self.loss_dice = build_loss(loss_dice)
            self.loss_rank = build_loss(loss_rank)

            self.mse_loss = nn.MSELoss()

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
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
                    gt_masks_list, img_metas):
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
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
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
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    def get_targets_sor(self, cls_scores_list, mask_preds_list, gt_labels_list,
                        gt_masks_list, img_metas):
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
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_sor, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_sor(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
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
        num_gts = gt_masks.shape[0]

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
        assign_result = self.sor_assigner.assign(cls_score, mask_points_pred,
                                                 gt_labels, gt_points_masks,
                                                 img_metas)
        sampling_result = self.sor_sampler.sample(assign_result, mask_pred,
                                                  gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label target
        labels = gt_labels.new_full((self.max_salient_object, ),
                                    self.max_ranks,
                                    dtype=torch.float32)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.max_salient_object, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.max_salient_object, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds', 'all_rank_scores', 'pred_salient_masks'))
    def loss(self, all_cls_scores, all_mask_preds, all_rank_scores, pred_salient_masks,
             gt_labels_list, gt_masks_list, rank_labels_list, img_metas, img):
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
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        img_list = [img for _ in range(num_dec_layers)]

        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        all_rank_labels_list = [rank_labels_list for _ in range(num_dec_layers)]
        losses_rank, = multi_apply(
            self.loss_node, all_rank_scores, pred_salient_masks,
            all_rank_labels_list, all_gt_masks_list, img_metas_list, img_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        loss_dict['loss_rank'] = losses_rank[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_rank_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_rank[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            loss_dict[f'd{num_dec_layer}.loss_rank'] = loss_rank_i
            num_dec_layer += 1
        return loss_dict

    # def loss_node(self, rank_scores, mask_preds, rank_labels, gt_masks_list, img_metas, img):
    #
    #     num_imgs = rank_scores.size(0)
    #     rank_scores_list = [rank_scores[i] for i in range(num_imgs)]
    #     mask_preds_list = [mask_preds[i] for i in range(num_imgs)]       # batch to list
    #     (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
    #      num_total_pos,
    #      num_total_neg) = self.get_targets_sor(rank_scores_list, mask_preds_list,
    #                                           rank_labels, gt_masks_list,
    #                                           img_metas)
    #     loss_rank = self.loss_rank(labels_list, rank_scores)
    #     return loss_rank,

    def get_high_fre_label(self, mask):
        mask = mask.unsqueeze(1).to(torch.float32)
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).cuda()
        conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel.shape, padding=1, stride=1, bias=False)
        kernel = kernel.reshape((1, 1, kernel.shape[0], kernel.shape[1]))
        conv2d.weight.data = kernel
        mask = conv2d(mask).detach()
        mask = F.pad(mask, pad=[1, 1, 1, 1], mode='reflect')
        mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=0)
        mask[mask > 0] = 1
        return mask.detach().squeeze(1)

    def get_low_fre_label(self, mask_targets_list, img):
        batch_size = len(mask_targets_list)
        low_fre = []
        for i in range(batch_size):
            for target in mask_targets_list[i]:
                low_fre_target = target * img[i]
                low_fre_target = low_fre_target.unsqueeze(0)
                low_fre.append(low_fre_target)
        low_fre = torch.cat(low_fre, dim=0) if len(low_fre) > 0 else low_fre
        return low_fre

    def loss_node(self, rank_scores, mask_preds, rank_labels, gt_masks_list, img_metas, img):

        num_imgs = rank_scores['rank_pred'].size(0)
        rank_scores_list = [rank_scores['rank_pred'][i] for i in range(num_imgs)]

        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]       # batch to list
        img = [img[i] for i in range(num_imgs)]

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets_sor(rank_scores_list, mask_preds_list,
                                              rank_labels, gt_masks_list,
                                              img_metas)
        loss_rank = self.loss_rank(labels_list, rank_scores)

        return loss_rank,

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
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
        num_imgs = cls_scores.size(0) # 就是batch_size
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)] #array变list
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
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
        loss_cls = self.loss_cls( # TODO
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())
        # loss_cls = self.loss_cls(
        #     cls_scores,
        #     labels)

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

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

        return loss_cls, loss_mask, loss_dice

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
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
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        cls_pred = self.cls_embed(decoder_out)

        # shape (batch_size, num_queries, 1024)
        node_embed = self.node_embed(decoder_out)

        # shape (batch_size, num_queries, c)
        mask_embed = self.mask_embed(decoder_out)

        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h*w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()
        return cls_pred, mask_pred, attn_mask, node_embed

    def forward_mask2former(self, feats, img_metas):
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
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        node_embed_list = []
        cls_pred, mask_pred, attn_mask, node_embed = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)
        node_embed_list.append(node_embed)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask, node_embed = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
            node_embed_list.append(node_embed)
        return cls_pred_list, mask_pred_list, node_embed_list, mask_features  # TODO

    def forward(self, feats, img_metas, img):
        cls_pred_list, mask_pred_list, node_embed_list, mask_features = self.forward_mask2former(feats, img_metas)  # TODO

        len_dec_layers = len(cls_pred_list)
        all_rank_scores = []
        pred_salient_masks = []
        all_salient_objs = []
        for i in range(len_dec_layers):
            cls_pred_last, mask_pred_last, node_embed_last = cls_pred_list[i], mask_pred_list[i], node_embed_list[i]
            # (batch_size, num_queries, cls_out_channels) (batch_size, num_queries, h, w)

            batch_salient_node = []
            batch_salient_mask = []
            batch_salient_indx = []
            for cls_pred, mask_pred, node_embed in zip(cls_pred_last, mask_pred_last, node_embed_last):

                scores = F.softmax(cls_pred, dim=-1)[:, :-1]

                scores_per_image, top_indices = scores.flatten(0, 1).topk(
                self.max_salient_object, sorted=False)

                salient_obj = torch.zeros(self.max_salient_object)
                salient_obj[scores_per_image > 0.80] = 1
                batch_salient_indx.append(salient_obj)

                salient_node = node_embed[top_indices]   #TODO
                salient_mask = mask_pred[top_indices]
                batch_salient_node.append(salient_node)
                batch_salient_mask.append(salient_mask)
            batch_salient_node = torch.stack(batch_salient_node, dim=0)
            batch_salient_mask = torch.stack(batch_salient_mask, dim=0)
            batch_salient_indx = torch.stack(batch_salient_indx, dim=0)

            rank_score = self.rank_head(batch_salient_node, batch_salient_mask, img)

            all_rank_scores.append(rank_score)
            pred_salient_masks.append(batch_salient_mask)
            all_salient_objs.append(batch_salient_indx)
        return cls_pred_list, mask_pred_list, all_rank_scores, pred_salient_masks, all_salient_objs


    def forward_train(self,
                      feats,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_ranks, #TODO
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
        all_cls_scores, all_mask_preds, all_rank_scores, pred_salient_masks, _ = self(feats, img_metas, img)
        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_semantic_seg, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, all_rank_scores, pred_salient_masks,
                           gt_labels, gt_masks, gt_ranks, img_metas, img)

        return losses

    def simple_test(self, feats, img_metas, img, **kwargs):
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
        all_cls_scores, all_mask_preds, all_rank_scores, pred_salient_masks, salient_idx = self(feats, img_metas, img)
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        mask_salient_idx = salient_idx[-1]

        mask_rank_results = all_rank_scores[-1]
        mask_rank_results = mask_rank_results['rank_pred_lf']
        mask_salient_results = pred_salient_masks[-1]
        # upsample masks
        img_shape = img_metas[0]['batch_input_shape']
        mask_salient_results = F.interpolate(
            mask_salient_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        # if len(torch.unique(mask_salient_idx)) == 1:
        #     return mask_rank_results, mask_salient_results
        #
        # sal_rank = []
        # sal_mask = []
        # for idx, sal in enumerate(mask_salient_idx.flatten(0, 1)):
        #     if sal == 1:
        #         sal_rank.append(mask_rank_results[:, idx])
        #         sal_mask.append(mask_salient_results[:, idx, ...])
        #
        # sal_rank = torch.stack(sal_rank, dim=1)
        # sal_mask = torch.stack(sal_mask, dim=1)

        return mask_rank_results, mask_salient_results          # 测试时返回的是预测的得分最高的目标以及分割结果


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


from torch.autograd import Variable
import numpy as np
from ..utils import SinePositionalEncoding, DetrTransformerDecoder

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
class DomainSepModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, norm_cfg=dict(type='GN', num_groups=32)):
        super(DomainSepModule, self).__init__()

        self.in_dim = in_dim

        self.hf_input_proj = Conv2d(5, 256, kernel_size=1)
        self.lf_input_proj = Conv2d(3*5, 256, kernel_size=1)

        self.hf_embed = nn.Embedding(1, 256)
        self.hf_query_embed = nn.Embedding(5, 256)  # (num_querys, feat_channels)

        self.lf_embed = nn.Embedding(1, 256)
        self.lf_query_embed = nn.Embedding(5, 256)  # (num_querys, feat_channels)

        self.hf_positional_encoding = SinePositionalEncoding(num_feats=128, normalize=True)
        self.hf_transformer_decoder = DetrTransformerDecoder(
            return_intermediate=False,
            num_layers=2,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None
        )

        self.lf_positional_encoding = SinePositionalEncoding(num_feats=128, normalize=True)
        self.lf_transformer_decoder = DetrTransformerDecoder(
            return_intermediate=False,
            num_layers=2,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None
        )

        self.hf_rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.lf_rank_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.shared_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.hf_private_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.lf_private_gnn_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.shared_gnn_dec = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)

        self.rank_head_hf = nn.Linear(in_dim, 1)
        self.rank_head_lf = nn.Linear(in_dim, 1)

        self.rank_embed = LinearBlock(in_dim*2, in_dim, drop=0., norm_cfg=norm_cfg)
        self.restore_hf = LinearBlock(in_dim*2, in_dim, drop=0., norm_cfg=norm_cfg)
        self.restore_lf = LinearBlock(in_dim*2, in_dim, drop=0., norm_cfg=norm_cfg)

        self.final_enc = GNNEncoder(in_dim=in_dim, num_classes=num_classes, norm_cfg=norm_cfg)
        self.rank_head_final = nn.Linear(in_dim, 1)

        self._init_weights()

    def _init_weights(self):
        self.rank_head_lf.weight.data.normal_(0, 0.02)
        self.rank_head_hf.weight.data.normal_(0, 0.02)
        self.rank_head_final.weight.data.normal_(0, 0.02)

    def forward(self, node_feature, mask_feature, img):
        '''

        Args:
            node_feature: [batch, num_nodes, c]
            mask_feature: [batch, num_nodes, h, w]
            img: [batch, 3, 4*h, 4*w]

        Returns: dict results

        '''
        batch_size = node_feature.size(0)

        # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
        hf_input = self.hf_input_proj(mask_feature)
        hf_input = hf_input.flatten(2).permute(2, 0, 1)
        hf_embed = self.hf_embed.weight[0].view(1, 1, -1)
        hf_input = hf_input + hf_embed

        # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
        hf_mask = hf_input.new_zeros(
            (batch_size,) + mask_feature.shape[-2:],
            dtype=torch.bool)

        hf_positional_encoding = self.hf_positional_encoding(
            hf_mask)
        hf_positional_encoding = hf_positional_encoding.flatten(
            2).permute(2, 0, 1)

        hf_query_embed = self.hf_query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        hf_node_feat = self.hf_transformer_decoder(
            query=node_feature.permute(1, 0, 2),
            key=hf_input,
            value=hf_input,
            query_pos=hf_query_embed,
            key_pos=hf_positional_encoding,
            attn_masks=None,
            query_key_padding_mask=None,
            # here we do not apply masking on padded region
            key_padding_mask=None
        )

        hf_node_feat = hf_node_feat.squeeze(0).permute(1, 0, 2)

        img = F.interpolate(img, scale_factor=0.25, mode='bilinear', align_corners=False)
        batch_lf_objs = []
        for i in range(batch_size):
            im = img[i]
            mask = mask_feature[i]
            lf_objs = []
            for obj in mask:
                attn_mask = torch.zeros_like(obj)
                attn_mask[obj.sigmoid() > 0.5] = 1
                lf_obj = attn_mask * im
                lf_objs.append(lf_obj)
            lf_objs = torch.stack(lf_objs, dim=0)
            batch_lf_objs.append(lf_objs)
        batch_lf_objs = torch.stack(batch_lf_objs, dim=0)
        batch_lf_objs = batch_lf_objs.reshape(batch_size, -1, batch_lf_objs.size(-2), batch_lf_objs.size(-1))


        # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
        lf_input = self.lf_input_proj(batch_lf_objs)
        lf_input = lf_input.flatten(2).permute(2, 0, 1)
        lf_embed = self.lf_embed.weight[0].view(1, 1, -1)
        lf_input = lf_input + lf_embed

        # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
        lf_mask = lf_input.new_zeros(
            (batch_size,) + batch_lf_objs.shape[-2:],
            dtype=torch.bool)

        lf_positional_encoding = self.lf_positional_encoding(
            lf_mask)
        lf_positional_encoding = lf_positional_encoding.flatten(
            2).permute(2, 0, 1)

        lf_query_embed = self.lf_query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        lf_node_feat = self.lf_transformer_decoder(
            query=node_feature.permute(1, 0, 2),
            key=lf_input,
            value=lf_input,
            query_pos=lf_query_embed,
            key_pos=lf_positional_encoding,
            attn_masks=None,
            query_key_padding_mask=None,
            # here we do not apply masking on padded region
            key_padding_mask=None
        )

        lf_node_feat = lf_node_feat.squeeze(0).permute(1, 0, 2)

        hf_feature = self.hf_rank_gnn_enc(hf_node_feat)
        lf_feature = self.lf_rank_gnn_enc(lf_node_feat)

        hf_shared_feature = self.shared_gnn_enc(hf_feature)
        lf_shared_feature = self.shared_gnn_enc(lf_feature)

        hf_private_feature = self.hf_private_gnn_enc(hf_feature)
        lf_private_feature = self.lf_private_gnn_enc(lf_feature)

        restored_hf_feature = self.restore_hf(torch.cat([hf_private_feature, hf_shared_feature], dim=-1))
        restored_lf_feature = self.restore_lf(torch.cat([lf_private_feature, lf_shared_feature], dim=-1))
        restored_hf_feature = self.shared_gnn_dec(restored_hf_feature)  # TODO: concat
        restored_lf_feature = self.shared_gnn_dec(restored_lf_feature)

        rank_pred_lf = self.rank_head_hf(lf_shared_feature).squeeze(-1)
        rank_pred_hf = self.rank_head_lf(hf_shared_feature).squeeze(-1)

        shared_feature = self.rank_embed(torch.cat([lf_shared_feature, hf_shared_feature], dim=-1))
        pred_feature = self.final_enc(shared_feature)
        rank_pred = self.rank_head_final(pred_feature).squeeze(-1)

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
                       rank_pred=rank_pred
                       )

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


