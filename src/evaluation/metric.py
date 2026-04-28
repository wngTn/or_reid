import logging

import torch
import numpy as np
import torch.nn.functional as F

from utils import is_tensor

LOGGER = logging.getLogger(__name__)


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = torch.from_numpy(msk1).cuda()
    if not is_tensor(msk2):
        msk2 = torch.from_numpy(msk2).cuda()
    n = msk1.size(0)
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).float()
    miou = inter.view(n, -1).sum(-1) / (union.view(n, -1).sum(-1) + eps)
    return miou


def evaluate_many_multi_view(distmat, q_pids, g_pids, q_sequences, metric, max_rank=1):
    """
    Multi-View Evaluation.

    Instead of one sample having a simple 1-D matching array, we have a 2-D array, for instance:
    Q_ID: {person_id}_{recording}_{sequence}
    Matches: [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # View 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # View 2
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # View 3
    ]
    We then reduce the matching array to a single 1-D array by taking the element that appears
    the most (multi-view voting).

    (q|g)_sequences: indicates to which sequence the sample belongs to.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        LOGGER.warning("Number of gallery samples is smaller than max_rank: got %s.", num_g)
    if metric in ["euc", "cos"]:
        indices = np.argsort(distmat.cpu().numpy(), axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    else:
        indices = np.argsort(-distmat, axis=1)
        preds = g_pids[indices]
        matches = (preds == q_pids[..., None].repeat(indices.shape[-1], -1)).astype(np.int32)

    all_cmc = []
    all_AP = []
    all_INP = []
    total_correct = 0
    num_valid_q = 0.

    id_accuracies = {}

    for q_seq in np.unique(q_sequences):
        q_idxs = np.where(q_sequences == q_seq)[0]
        q_pid = q_pids[q_idxs]
        pid = q_pid[0]
        assert (q_pid == pid).all(), "All samples in the same sequence should have the same PID"

        multi_view_orig_cmc = matches[q_idxs]

        if metric == "euc":
            _multi_view_orig_cmc = np.zeros_like(multi_view_orig_cmc)
            first_one_indices = np.argmax(multi_view_orig_cmc == 1, axis=1)
            _multi_view_orig_cmc[np.arange(multi_view_orig_cmc.shape[0]), first_one_indices] = 1
            last_one_indices = np.max(np.where(_multi_view_orig_cmc == 1, np.arange(multi_view_orig_cmc.shape[1]), -1), axis=1)
            max_rank = np.max([np.max(last_one_indices), len(np.unique(g_pids))])
            multi_view_orig_cmc = _multi_view_orig_cmc[:, :max_rank + 1]

        multi_view_cumsum = np.cumsum(multi_view_orig_cmc, axis=1)
        _orig_cmc = np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), axis=0, arr=multi_view_cumsum)
        orig_cmc = np.zeros_like(_orig_cmc)
        try:
            orig_cmc[np.where(_orig_cmc == 1)[0][0]] = 1
        except IndexError:
            pass

        if not np.any(orig_cmc):
            all_INP.append(0)
            all_cmc.append(np.zeros(3))
            all_AP.append(0)
            if pid not in id_accuracies:
                id_accuracies[pid] = {"total_correct": 0, "total_predictions": 0}
            id_accuracies[pid]["total_predictions"] += 1
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:3])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        if pid not in id_accuracies:
            id_accuracies[pid] = {"total_correct": 0, "total_predictions": 0}
        id_accuracies[pid]["total_correct"] += orig_cmc[0]
        id_accuracies[pid]["total_predictions"] += 1

        total_correct += orig_cmc[0]

    for pid in id_accuracies:
        id_accuracies[pid] = id_accuracies[pid]["total_correct"] / id_accuracies[pid]["total_predictions"]

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    accuracy = total_correct / len(np.unique(q_sequences))

    return all_cmc, mAP, mINP, accuracy, id_accuracies
