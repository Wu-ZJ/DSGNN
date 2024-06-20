import torch
import torch.nn as nn
from itertools import combinations
from ..builder import LOSSES


def saliency_ranking_loss(rank_labels,
                          saliency_score):
    N = len(rank_labels)
    saliency_score = saliency_score.reshape(-1)
    S1, S2 = torch.meshgrid((saliency_score, saliency_score))
    S = -S1 + S2
    R1, R2 = torch.meshgrid((rank_labels, rank_labels))
    R = (R1 - R2).cuda()
    R[R > 0] = 1
    R[R < 0] = -1
    S = S * R
    S = torch.log(1+torch.exp(S))
    S[R == 0] = 0
    S = torch.triu(S, 1)
    B = torch.abs((R1 - R2).cuda().float())
    Wr_m = torch.sum(torch.arange(1, N) * torch.arange(N - 1, 0, -1)).float()
    B = B / Wr_m
    S = S * B
    relation_loss = torch.sum(S)
    return relation_loss


@LOSSES.register_module()
class RelationLoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0):
        super(RelationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_sigmoid = False
        # self.sim_loss = nn.CosineEmbeddingLoss()

    def forward(self, gt_ranks, saliency_scores):
        """
        gt_ranks: list
        salienct_scores: list
        """
        # sim_loss = self.sim_loss(
        #     gt_ranks.contiguous().view(-1, gt_ranks.size(-1)),
        #     saliency_scores.contiguous().view(-1, saliency_scores.size(-1)),
        #     target=torch.tensor([1]).cuda())

        relation_losses = 0
        for gt_ranks_per_image, score_per_image in zip(gt_ranks, saliency_scores):
            if len(gt_ranks_per_image) > 1:
                relation_loss = self.loss_compute(gt_ranks_per_image, score_per_image)
            else:
                relation_loss = 0
            relation_losses += relation_loss

        # return self.loss_weight * (relation_losses/len(gt_ranks) + 0.5*sim_loss)
        return self.loss_weight * (relation_losses / len(gt_ranks))

    def loss_compute(self, rank_labels, saliency_score):
        N = len(rank_labels)
        saliency_score = saliency_score.reshape(-1)
        S1, S2 = torch.meshgrid((saliency_score, saliency_score))
        S = -S1 + S2
        R1, R2 = torch.meshgrid((rank_labels, rank_labels))
        R = (R1 - R2).cuda()
        R[R > 0] = 1
        R[R < 0] = -1
        S = S * R
        S = torch.log(1+torch.exp(S))
        S[R == 0] = 0
        S = torch.triu(S, 1)
        B = torch.abs((R1 - R2).cuda().float())
        Wr_m = torch.sum(torch.arange(1, N) * torch.arange(N - 1, 0, -1)).float()
        B = B / Wr_m
        S = S * B
        relation_loss = torch.sum(S)
        return relation_loss


class PearsonEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.):
        super(PearsonEmbeddingLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, target, reduction='mean'):
        batch_size = x1.size(0)
        if len(target) == 1:
            target = target.repeat(batch_size)

        assert x1.shape == x2.shape
        if len(x1.shape) == 3:
            x1 = x1.contiguous().view(-1, x1.size(-1))
            x2 = x2.contiguous().view(-1, x2.size(-1))

        scores = []
        for i in range(batch_size):
            score = self._pearson_similarity(x1[i], x2[i])
            score = self._cal_score(score, target[i].item())
            scores.append(score)
        scores = torch.stack(scores, 0)
        if reduction == 'mean':
            return scores.mean()
        elif reduction == 'sum':
            return scores.sum()

    def _pearson_similarity(self, x, y):
        n = len(x)
        # simple sums
        sum1 = sum(float(x[i]) for i in range(n))
        sum2 = sum(float(y[i]) for i in range(n))
        # sum up the squares
        sum1_pow = sum([pow(v, 2.0) for v in x])
        sum2_pow = sum([pow(v, 2.0) for v in y])
        # sum up the products
        p_sum = sum([x[i] * y[i] for i in range(n)])
        # 分子num，分母den
        num = p_sum - (sum1 * sum2 / n)
        den = torch.sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
        if den == 0:
            return 0.0
        return num / den

    def _cal_score(self, score, target):
        if target == 1:
            return 1 - score
        else:
            return max(0, score - self.margin)


@LOSSES.register_module()
class DomainLoss(nn.Module):
    def __init__(self,
                 diff_weight=1,
                 # sim_weight=0.01,
                 recon_weight=0.5,
                 rank_weight1=1,
                 rank_weight2=1,
                 rank_weight3=5,
                 selfsup_weight=0.1,
                 loss_weight=1.0):
        super(DomainLoss, self).__init__()

        self.diff_weight = diff_weight
        # self.sim_weight = sim_weight
        self.recon_weight = recon_weight
        self.rank_weight1 = rank_weight1
        self.rank_weight2 = rank_weight2
        self.rank_weight3 = rank_weight3
        self.selfsup_weight = selfsup_weight

        self.rank_loss = RelationLoss()
        self.recon_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, gt_ranks, results):

        hf_feature = results['hf_feature']
        lf_feature = results['lf_feature']

        hf_shared_feature = results['hf_shared_feature']
        lf_shared_feature = results['lf_shared_feature']
        hf_private_feature = results['hf_private_feature']
        lf_private_feature = results['lf_private_feature']
        restored_hf_feature = results['restored_hf_feature']
        restored_lf_feature = results['restored_lf_feature']
        rank_pred_hf = results['rank_pred_hf']
        rank_pred_lf = results['rank_pred_lf']
        rank_pred = results['rank_pred']
        pred_feature = results['pred_feature']

        diff_loss1 = self._diff_loss(hf_shared_feature, lf_shared_feature)

        diff_loss2 = self._diff_loss(hf_shared_feature, hf_private_feature)
        diff_loss3 = self._diff_loss(lf_shared_feature, lf_private_feature)

        recon_loss_hf = self.recon_loss(restored_hf_feature, hf_feature)
        recon_loss_lf = self.recon_loss(restored_lf_feature, lf_feature)

        rank_loss_final1 = self.rank_loss(gt_ranks, rank_pred_hf)
        rank_loss_final2 = self.rank_loss(gt_ranks, rank_pred_lf)
        rank_loss = self.rank_loss(gt_ranks, rank_pred)

        loss = self.diff_weight * (diff_loss2 + diff_loss3) + \
               self.diff_weight * diff_loss1 + \
               self.recon_weight * (recon_loss_hf + recon_loss_lf) + \
               self.rank_weight2 * (rank_loss_final1 + rank_loss_final2) + \
               self.rank_weight3 * rank_loss

        return loss

    def _diff_loss(self, private_samples, shared_samples, weight=1):
        """

        Args:
            private_samples: shape:[batchsize, num_nodes, c]
            shared_samples: shape:[batchsize, num_nodes, c]
            weight: loss weight

        Returns: difference loss

        """
        private_samples = private_samples - torch.mean(private_samples, 0)
        shared_samples = shared_samples - torch.mean(shared_samples, 0)
        private_samples = torch.nn.functional.normalize(private_samples, p=2, dim=1)
        shared_samples = torch.nn.functional.normalize(shared_samples, p=2, dim=1)
        correlation_matrix = torch.matmul(private_samples.transpose(1, 2), shared_samples)
        cost = torch.mean(torch.square(correlation_matrix)) * weight
        cost = torch.where(cost > torch.tensor(0, dtype=torch.float32).to(cost.device),
                           cost, torch.tensor(0, dtype=torch.float32).to(cost.device))
        return cost

    def _mean_pairwise_squared_error(self, labels, predictions, weights=1.0):
        labels = labels.to(torch.float32)
        predictions = predictions.to(torch.float32)
        weights = torch.tensor(weights).to(torch.float32).to(labels.device)

        assert labels.shape == predictions.shape, 'labels and predictions must have the same shape!'

        diffs = torch.subtract(predictions, labels)
        reduction_indices = torch.arange(1, diffs.dim()).tolist()

        sum_squares_per_batch = torch.sum(torch.square(diffs), dim=reduction_indices, keepdim=True)
        num_present_per_batch = torch.tensor(diffs.numel()).to(labels.device)

        term1 = 2.0 * torch.div(sum_squares_per_batch, num_present_per_batch - 1)

        sum_diff = torch.sum(diffs, dim=reduction_indices, keepdim=True)
        term2 = 2.0 * torch.div(torch.square(sum_diff), torch.multiply(num_present_per_batch, num_present_per_batch - 1))

        weighted_losses = torch.multiply(term1 - term2, weights)
        loss = torch.sum(weighted_losses)

        mean_loss = torch.where(
            torch.sum(num_present_per_batch) > torch.tensor(0, dtype=torch.float32).to(labels.device),
            loss,
            torch.zeros_like(loss).to(labels.device))
        return mean_loss

    def _gnn_sep_loss(self, results, gt_rank):
        shared_feature = results['shared_feature']
        private_feature = results['private_feature']
        restored_feature = results['restored_feature']
        node_feature = results['node_feature']

        shared_feats = []
        private_feats = []
        restored_feats = []
        node_feats = []
        for gt, sf, pf, rf, nf in zip(gt_rank, shared_feature, private_feature, restored_feature, node_feature):
            _, indice = gt.topk(len(gt), sorted=True)

            new_gt = [i for i in gt if i > 0]
            saved_length = len(new_gt) if len(new_gt) > 0 else len(gt)

            sf = sf[indice]
            sf = sf[: saved_length]

            pf = pf[indice]
            pf = pf[: saved_length]

            rf = rf[indice]
            rf = rf[: saved_length]

            nf = nf[indice]
            nf = nf[: saved_length]

            shared_feats.append(sf)
            private_feats.append(pf)
            restored_feats.append(rf)
            node_feats.append(nf)

        index = len(shared_feats[0]) if len(shared_feats[0]) <= len(shared_feats[1]) else len(shared_feats[1])
        sim_losses = []
        diff_losses_1 = []
        diff_losses_2 = []
        recon_losses_1 = []
        recon_losses_2 = []
        for i in range(index):
            sim_loss = self.recon_loss(shared_feats[0][i], shared_feats[1][i])
            sim_losses.append(sim_loss)

            diff_loss_1 = self._diff_loss(private_feats[0][i].unsqueeze(0).unsqueeze(0), shared_feats[0][i].unsqueeze(0).unsqueeze(0))
            diff_losses_1.append(diff_loss_1)

            diff_loss_2 = self._diff_loss(private_feats[1][i].unsqueeze(0).unsqueeze(0), shared_feats[1][i].unsqueeze(0).unsqueeze(0))
            diff_losses_2.append(diff_loss_2)

            recon_loss_1 = self.recon_loss(restored_feats[0][i], node_feats[0][i])
            recon_losses_1.append(recon_loss_1)

            recon_loss_2 = self.recon_loss(restored_feats[1][i], node_feats[1][i])
            recon_losses_2.append(recon_loss_2)

        loss = (sum(sim_losses) + sum(diff_losses_1) + sum(diff_losses_2) + sum(recon_losses_1) + sum(recon_losses_2)) / index
        return loss

    def _gnn_sim_loss(self, pred_feature, gt_rank):

        shared_feats = []
        for gt, sf in zip(gt_rank, pred_feature):
            _, indice = gt.topk(len(gt), sorted=True)

            new_gt = [i for i in gt if i > 0]
            saved_length = len(new_gt) if len(new_gt) > 0 else len(gt)

            sf = sf[indice]
            sf = sf[: saved_length]

            shared_feats.append(sf)

        index = len(shared_feats[0]) if len(shared_feats[0]) <= len(shared_feats[1]) else len(shared_feats[1])
        sim_losses = []
        for i in range(index):
            sim_loss = self.recon_loss(shared_feats[0][i], shared_feats[1][i])
            sim_losses.append(sim_loss)

        loss = sum(sim_losses) / index
        return loss

    def node_selfsup_loss(self, x, gt_rank):
        """

        Args:
            x: shared feature, shape: [B, N, C]
            gt_rank: shape: [B, N]
        Returns:

        """
        feats = []
        for gt, feat in zip(gt_rank, x):
            _, indice = gt.topk(len(gt), sorted=True)
            feat = feat[indice]
            feats.append(feat)

        l = [i for i in range(len(feats))]
        combs = list(combinations(l, 2))

        losses = []
        for comb in combs:
            q = feats[comb[0]]
            k = feats[comb[1]]

            logits = torch.einsum('nc, kc -> nk', q, k)
            labels = torch.tensor([i for i in range(len(q))], dtype=torch.int64).to(logits.device)
            loss = self.ce_loss(logits, labels)
            losses.append(loss)
        loss_ss = torch.tensor(losses).sum() / len(combs)

        return loss_ss


@LOSSES.register_module()
class DomainSepLoss(nn.Module):
    def __init__(self,
                 diff_weight=1,
                 sim_weight=2,
                 recon_weight=0.05,
                 loss_weight=1.0):
        super(DomainSepLoss, self).__init__()

        self.diff_weight = diff_weight
        self.sim_weight = sim_weight
        self.recon_weight = recon_weight

        # self.sim_loss = PearsonEmbeddingLoss()
        self.sim_loss = nn.CosineEmbeddingLoss()
        self.pearson_loss = PearsonEmbeddingLoss()
        self.recon_loss = torch.nn.MSELoss()

    def forward(self, results, gt_rank):

        hf_feature = results['hf_feature']
        lf_feature = results['lf_feature']

        hf_shared_feature = results['hf_shared_feature']
        lf_shared_feature = results['lf_shared_feature']
        hf_private_feature = results['hf_private_feature']
        lf_private_feature = results['lf_private_feature']
        restored_hf_feature = results['restored_hf_feature']
        restored_lf_feature = results['restored_lf_feature']

        gnn_sep_results = results['gnn_sep_results']
        gnn_sep_loss = self._gnn_sep_loss(gnn_sep_results, gt_rank)

        # sim_loss = self.sim_loss(
        #     hf_shared_feature.contiguous().view(-1, hf_shared_feature.size(-1)),
        #     lf_shared_feature.contiguous().view(-1, lf_shared_feature.size(-1)),
        #     target=torch.tensor([1]).cuda())
        sim_loss = self.pearson_loss(hf_shared_feature, lf_shared_feature, target=torch.tensor([1]).cuda())

        diff_loss2 = self._diff_loss(hf_shared_feature, hf_private_feature)
        diff_loss3 = self._diff_loss(lf_shared_feature, lf_private_feature)

        recon_loss_hf = self.recon_loss(restored_hf_feature, hf_feature)
        recon_loss_lf = self.recon_loss(restored_lf_feature, lf_feature)

        loss = self.sim_weight * sim_loss + \
               self.diff_weight * (diff_loss2 + diff_loss3) +\
               self.recon_weight * (recon_loss_hf + recon_loss_lf) + \
               gnn_sep_loss
        return loss

    def _diff_loss(self, private_samples, shared_samples, weight=1):
        """

        Args:
            private_samples: shape:[batchsize, num_nodes, c]
            shared_samples: shape:[batchsize, num_nodes, c]
            weight: loss weight

        Returns: difference loss

        """
        private_samples = private_samples - torch.mean(private_samples, 0)
        shared_samples = shared_samples - torch.mean(shared_samples, 0)
        private_samples = torch.nn.functional.normalize(private_samples, p=2, dim=1)
        shared_samples = torch.nn.functional.normalize(shared_samples, p=2, dim=1)
        correlation_matrix = torch.matmul(private_samples.transpose(1, 2), shared_samples)
        cost = torch.mean(torch.square(correlation_matrix)) * weight
        cost = torch.where(cost > torch.tensor(0, dtype=torch.float32).to(cost.device),
                           cost, torch.tensor(0, dtype=torch.float32).to(cost.device))
        return cost

    def _mean_pairwise_squared_error(self, labels, predictions, weights=1.0):
        labels = labels.to(torch.float32)
        predictions = predictions.to(torch.float32)
        weights = torch.tensor(weights).to(torch.float32).to(labels.device)

        assert labels.shape == predictions.shape, 'labels and predictions must have the same shape!'

        diffs = torch.subtract(predictions, labels)
        reduction_indices = torch.arange(1, diffs.dim()).tolist()

        sum_squares_per_batch = torch.sum(torch.square(diffs), dim=reduction_indices, keepdim=True)
        num_present_per_batch = torch.tensor(diffs.numel()).to(labels.device)

        term1 = 2.0 * torch.div(sum_squares_per_batch, num_present_per_batch - 1)

        sum_diff = torch.sum(diffs, dim=reduction_indices, keepdim=True)
        term2 = 2.0 * torch.div(torch.square(sum_diff), torch.multiply(num_present_per_batch, num_present_per_batch - 1))

        weighted_losses = torch.multiply(term1 - term2, weights)
        loss = torch.sum(weighted_losses)

        mean_loss = torch.where(
            torch.sum(num_present_per_batch) > torch.tensor(0, dtype=torch.float32).to(labels.device),
            loss,
            torch.zeros_like(loss).to(labels.device))
        return mean_loss

    def _gnn_sep_loss(self, results, gt_rank):
        sep_node = results['sep_node']
        noi_node = results['noi_node']

        sep_feats = []
        noi_feats = []
        for gt, sf, nf in zip(gt_rank, sep_node, noi_node):
            _, indice = gt.topk(len(gt), sorted=True)

            new_gt = [i for i in gt if i > 0]
            saved_length = len(new_gt) if len(new_gt) > 0 else len(gt)

            sf = sf[indice]
            sf = sf[: saved_length]

            nf = nf[indice]
            nf = nf[: saved_length]

            sep_feats.append(sf)
            noi_feats.append(nf)

        index = len(sep_feats[0]) if len(sep_feats[0]) <= len(sep_feats[1]) else len(sep_feats[1])
        sim_losses = []
        dif_losses = []
        for i in range(index):
            # sim_loss = self.recon_loss(sep_feats[0][i], sep_feats[1][i])
            sim_loss = self.sim_loss(sep_feats[0][i].unsqueeze(0), sep_feats[1][i].unsqueeze(0), target=torch.tensor([1]).cuda())
            sim_losses.append(sim_loss)

            dif_loss = self._diff_loss(noi_feats[0][i].unsqueeze(0).unsqueeze(0), noi_feats[1][i].unsqueeze(0).unsqueeze(0))
            dif_losses.append(dif_loss)

        loss = (sum(sim_losses) + sum(dif_losses)) / index
        return loss