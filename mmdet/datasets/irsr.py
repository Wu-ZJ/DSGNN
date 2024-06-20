import contextlib
import io
import cv2
import itertools
import json
import pickle
import logging
import scipy.stats as sc
import os.path
import os.path as osp
import pandas as pd
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from torch.utils.data import Dataset
from .custom import CustomDataset
from pycocotools import mask as coco_mask
from sklearn.metrics import mean_absolute_error


@DATASETS.register_module()
class IRSRDataset(CustomDataset):
    CLASSES = ('Salient Object',)
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228)]
    RANK_PIXEL = [255, 229, 204, 178, 153] + 100 * [153]
    # RANK_PIXEL = [153, 178, 204, 229, 255] + 100 * [255]
    SAL_VAL_THRESH = 0.5

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 ):
        super(IRSRDataset, self).__init__(
            ann_file,
            pipeline,
            classes,
            data_root,
            img_prefix,
            seg_prefix,
            seg_suffix,
            proposal_file,
            test_mode,
            filter_empty_gt,
            file_client_args)

    def load_annotations(self, ann_file):
        """

        """
        with open(ann_file, "rb") as a:
            info = pickle.load(a)

        data_infos = [{'filename': info[i]['file_name'].rsplit("/", 1)[1],
                       'object_data': info[i]['annotations'],
                       'width': 640,
                       'height': 480,
                       'rank': info[i]['rank']}
                      for i in range(len(info))]
        return data_infos

    def get_ann_info(self, idx):
        """
        Args:
            idx (int): Index of data.
        """
        object_data = self.data_infos[idx]['object_data']
        img_name = self.data_infos[idx]['filename'].rsplit('.', 1)[0]
        rank_labels = np.array(self.data_infos[idx]['rank'], dtype=np.float32)
        rank_labels = np.array([sorted(rank_labels).index(a) + 1 for a in rank_labels], dtype=np.float32)
        labels = np.array([0 for _ in range(len(rank_labels))], dtype=np.int64)
        # masks = [object_data[i]['segmentation'] for i in range(len(object_data))]
        masks = [obj['segmentation'] for obj in object_data]

        bboxes = []
        for i in range(len(object_data)):
            bbox_info = object_data[i]['bbox']
            bbox_mode = object_data[i]['bbox_mode']
            assert bbox_mode == 'xywh'
            if bbox_mode == 'xywh':
                bbox = [bbox_info[0], bbox_info[1], bbox_info[0]+bbox_info[2], bbox_info[1]+bbox_info[3]]
                bboxes.append(bbox)
            else:
                bboxes.append(bbox_info)

        bboxes = np.array(bboxes, dtype=np.float32)
        # bboxes = np.array([object_data[i]['bbox'] for i in range(len(object_data))], dtype=np.float32)
        seg_map = img_name + self.seg_suffix

        # print('first', len(rank_labels), len(masks), len(bboxes))
        assert len(rank_labels) == len(masks) == len(bboxes)
        ann_info = dict(
            bboxes=bboxes,
            labels=labels,
            rank_labels=rank_labels,
            masks=masks,
            seg_map=seg_map)

        return ann_info

    def calc_iou(self, mask_a, mask_b):
        mask_a = mask_a.astype(np.float32)
        mask_b = mask_b.astype(np.float32)
        intersection = (mask_a + mask_b >= 2).astype(np.float32).sum()
        iou = intersection / (mask_a + mask_b >= 1).astype(np.float32).sum()
        return iou

    def match(self, matrix, iou_thread):
        matched_gts = np.arange(matrix.shape[0])
        matched_ranks = matrix.argsort()[:, -1]
        for i, j in zip(matched_gts, matched_ranks):
            if matrix[i][j] < iou_thread:
                matched_ranks[i] = -1
        if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
            for i in set(matched_ranks):
                if i >= 0:
                    index_i = np.nonzero(matched_ranks == i)[0]
                    if len(index_i) > 1:
                        score_index = matched_ranks[index_i[0]]
                        ious = matrix[:, score_index][index_i]
                        max_index = index_i[ious.argsort()[-1]]
                        rm_index = index_i[np.nonzero(index_i != max_index)[0]]
                        matched_ranks[rm_index] = -1
        if len(set(matched_ranks[matched_ranks != -1])) < len(matched_ranks[matched_ranks != -1]):
            raise KeyError
        if len(matched_ranks) < matrix.shape[1]:
            for i in range(matrix.shape[1]):
                if i not in matched_ranks:
                    matched_ranks = np.append(matched_ranks, i)
        return matched_ranks

    def get_rank_index(self, gt_masks, segmaps, iou_thread, rank_scores):
        segmaps[segmaps > 0.5] = 1
        segmaps[segmaps <= 0.5] = 0

        ious = np.zeros([len(gt_masks), len(segmaps)])
        for i in range(len(gt_masks)):
            for j in range(len(segmaps)):
                ious[i][j] = self.calc_iou(gt_masks[i], segmaps[j])
        matched_ranks = self.match(ious, iou_thread)
        unmatched_index = np.argwhere(matched_ranks == -1).squeeze(1)
        matched_ranks = matched_ranks[matched_ranks >= 0]
        rank_scores = rank_scores[matched_ranks]
        rank_scores = rank_scores[:len(gt_masks)]
        rank_index = np.array([sorted(rank_scores).index(a) + 1 for a in rank_scores])

        for i in range(len(unmatched_index)):
            rank_index = np.insert(rank_index, unmatched_index[i], 0)
        rank_index = rank_index[:len(gt_masks)]
        return rank_index

    def evaluate_sa_sor(self, results, iou_thread):
        print('evaluate_sa_sor...')
        p_sum = 0
        num = len(results)
        for indx, result in enumerate(results):
            gt_masks = result['gt_masks']
            segmaps = result['segmaps']
            gt_ranks = result['gt_ranks']
            rank_scores = result['rank_scores']
            rank_scores = np.array(rank_scores)[:, None]

            if len(gt_ranks) == 1:
                num = num - 1
                continue

            gt_index = np.array([sorted(gt_ranks).index(a) + 1 for a in gt_ranks])
            if len(segmaps) == 0:
                rank_index = np.zeros_like(gt_ranks)
            else:
                rank_index = self.get_rank_index(gt_masks, segmaps, iou_thread, rank_scores)
            # print('gt_index: ', gt_index)
            # print('rank_index: ', rank_index)
            gt_index = pd.Series(gt_index)
            rank_index = pd.Series(rank_index)
            if rank_index.var() == 0:
                p = 0
            else:
                p = gt_index.corr(rank_index, method='pearson')
            if not np.isnan(p):
                p_sum += p
            else:
                num -= 1

        fianl_p = p_sum / num
        return fianl_p

    def evaluate_mae(self, results):
        print('evaluate_mae...')
        mae_results = []
        for result in results:
            gt_masks = result['gt_masks']
            segmaps = result['segmaps']
            names = result['names']
            gt_masks = np.stack(gt_masks, axis=0)

            # TODO:[n, h, w] -> [h, w]
            gt_ranks = (result['gt_ranks']) / len(result['gt_ranks'])
            pred_ranks = (result['rank_pred_level']) / len(result['rank_pred_level'])

            post_gt_masks = np.zeros((gt_masks.shape[1], gt_masks.shape[2]))
            post_pred_masks = np.zeros((segmaps.shape[1], segmaps.shape[2]))

            for i in range(len(segmaps)):
                post_pred_masks[segmaps[i] > 0.5] = pred_ranks[i]
            for i in range(len(gt_masks)):
                post_gt_masks[gt_masks[i] != 0] = gt_ranks[i]

            # cv2.imwrite(osp.join('/opt/data/private/mmdetection/work_dirs/test', names), post_pred_masks * 255)

            post_gt_masks = np.array(post_gt_masks.flatten()).astype(np.float32)
            post_pred_masks = np.array(post_pred_masks.flatten()).astype(np.float32)
            mae = mean_absolute_error(post_gt_masks, post_pred_masks)
            mae_results.append(mae)

        mae_results = sum(mae_results) / len(mae_results)
        return mae_results

    def get_usable_salient_objects_agreed(self, image_1_list, image_2_list):
        # Remove indices list
        rm_list = []
        for idx in range(len(image_1_list)):
            v = image_1_list[idx]
            v2 = image_2_list[idx]

            if v == 0 or v2 == 0:
                rm_list.append(idx)

        # Use indices list
        use_list = list(range(0, len(image_1_list)))
        use_list = list(np.delete(np.array(use_list), rm_list))

        # Remove the indices
        x = np.array(image_1_list)
        y = np.array(image_2_list)
        x = list(np.delete(x, rm_list))
        y = list(np.delete(y, rm_list))

        return x, y, use_list

    def calculate_spr(self, results, iou_thread):
        spr_data = []
        for indx, result in enumerate(results):
            gt_masks = result['gt_masks']  # list[(h, w), ...]
            segmaps = result['segmaps']  # array [n, h, w]
            gt_ranks = result['gt_ranks']  # list
            instance_pix_count = [np.sum(gt_masks[i].astype(np.float32)) for i in range(len(gt_masks))]

            gt_ranks = (len(gt_ranks) - gt_ranks) + 1
            rank_pred_level = (len(result['rank_pred_level']) - result['rank_pred_level']) + 1.

            rank_pred_level = np.array([i for i in rank_pred_level], dtype=np.int32)

            pred_sal_map = np.zeros((segmaps.shape[1], segmaps.shape[2]))
            for idx, rank in enumerate(rank_pred_level):
                pred_sal_map[segmaps[idx] > 0] = rank

            # Get corresponding predicted rank for each gt salient objects
            pred_ranks = []
            # Create mask for each salient object
            for s_i in range(len(gt_masks)):
                gt_seg_mask = gt_masks[s_i]
                gt_pix_count = instance_pix_count[s_i]

                pred_seg = np.where(gt_seg_mask == 1, pred_sal_map, 0)

                # number of pixels with predicted values
                pred_pix_loc = np.where(pred_seg > 0)

                pred_pix_num = len(pred_pix_loc[0])

                # Get rank of object
                r = 0
                if pred_pix_num > int(gt_pix_count * iou_thread):
                    vals = pred_seg[pred_pix_loc[0], pred_pix_loc[1]]
                    mode = sc.mode(vals)[0][0]
                    r = mode

                pred_ranks.append(r)
            # print('sor_gt_ranks: ', gt_ranks)
            # print('sor_pred_ranks: ', pred_ranks)

            # Remove objects with no saliency value in both list
            gt_ranks, pred_ranks, use_indices_list = \
                self.get_usable_salient_objects_agreed(gt_ranks, pred_ranks)

            spr = None

            if len(gt_ranks) > 1:
                spr = sc.spearmanr(gt_ranks, pred_ranks)
            elif len(gt_ranks) == 1:
                spr = 1

            d = [spr, use_indices_list]
            spr_data.append(d)
        return spr_data

    def extract_spr_value(self, data_list):
        use_idx_list = []
        spr = []
        for i in range(len(data_list)):
            s = data_list[i][0]

            if s == 1:
                spr.append(s)
                use_idx_list.append(i)
            elif s and not np.isnan(s[0]):
                spr.append(s[0])
                use_idx_list.append(i)

        return spr, use_idx_list

    def cal_avg_spr(self, data_list):
        spr = np.array(data_list)
        avg = np.average(spr)
        return avg

    def get_norm_spr(self, spr_value):
        #       m - r_min
        # m -> ---------------- x (t_max - t_min) + t_min
        #       r_max - r_min
        #
        # m = measure value
        # r_min = min range of measurement
        # r_max = max range of measurement
        # t_min = min range of desired scale
        # t_max = max range of desired scale

        r_min = -1
        r_max = 1

        norm_spr = (spr_value - r_min) / (r_max - r_min)

        return norm_spr

    def evaluate_sor(self, results, iou_thread):
        print('evaluate_sor...')
        spr_all_data = self.calculate_spr(results, iou_thread)

        spr_data, spr_use_idx = self.extract_spr_value(spr_all_data)

        avg_spr = self.cal_avg_spr(spr_data)
        avg_spr_norm = self.get_norm_spr(avg_spr)

        return avg_spr_norm

    def convert_coco_poly_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = np.any(mask, axis=2)
            masks.append(mask)
        return masks

    def evaluate(self,
                 results,
                 metric='mae',
                 logger=None,
                 iou_thr=0.5):
        """
        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['mae', 'sor', 'ssor']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        evaluate_data = []
        for i in range(len(annotations)):
            h, w = self.data_infos[i]['height'], self.data_infos[i]['width']
            gt_ranks = annotations[i]['rank_labels']
            gt_ranks = np.array([sorted(gt_ranks).index(a) + 1 for a in gt_ranks])
            data = dict(
                gt_masks=self.convert_coco_poly_mask(annotations[i]['masks'], h, w),
                segmaps=results[i]['rank_results']['mask_pred_binary'],
                gt_ranks=gt_ranks,
                rank_scores=results[i]['rank_results']['mask_score'],
                rank_pred_level=results[i]['rank_results']['rank_pred'],
                names=annotations[i]['seg_map'])
            evaluate_data.append(data)

        # evaluate_data = evaluate_data[:10]
        mae = self.evaluate_mae(evaluate_data)
        sor = self.evaluate_sor(evaluate_data, iou_thr)
        sa_sor = self.evaluate_sa_sor(evaluate_data, iou_thr)
        eval_results = dict(
            mae=mae,
            sor=sor,
            sa_sor=sa_sor)

        return eval_results





