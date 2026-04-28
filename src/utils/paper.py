from collections import defaultdict
import numpy as np
from pathlib import Path

def create_heatmap_nn_for_paper(distmat, q_pids, g_pids, q_sequences, modality):
    sorted_unique_ids, inverse_indices = np.unique(g_pids, return_inverse=True)
    per_person_heatmap = defaultdict(lambda: np.empty((0, len(sorted_unique_ids))))
    
    for q_seq in np.unique(q_sequences):
        # get query indices for the current sequence
        q_idxs = np.where(q_sequences == q_seq)[0]
        q_pid = q_pids[q_idxs]
        pid = q_pid[0]
        assert (q_pid == pid).all(), "All samples in the same sequence should have the same PID"

        # compute cmc curve
        distmat_views = distmat[q_idxs] # (num_query, len(g_pids))
        aggregated_distmat = np.zeros((distmat_views.shape[0], len(sorted_unique_ids)))
        # Step 3: Group columns by unique ID and compute the average using bincount
        for i in range(len(sorted_unique_ids)):
            aggregated_distmat[:, i] = np.mean(distmat_views[:, inverse_indices == i], axis=1)

        per_person_heatmap[pid] = np.concatenate((per_person_heatmap[pid], aggregated_distmat), axis=0)

    heatmap = np.zeros((len(per_person_heatmap), len(sorted_unique_ids)))
    for i, values in per_person_heatmap.items():
        heatmap[i] = np.mean(values, axis=0)

    output_path = Path("output", "paper", f"heatmap_nn_{modality}_mm_or.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, heatmap)
    print(f"Saved heatmap to {output_path}")
    return heatmap

def create_heatmap_svm_for_paper(distmat, q_pids, q_sequences, modality):
    """
    Multi-View Evaluation
    Instead of one sample having a simple 1-D matching array, we have a 2-D array, for instance:
    Q_ID: {person_id}_{recording}_{sequence}
    Matches: [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # View 1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # View 2
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # View 3
    ]
    We then reduce the matching array to a single 1-D array by taking the element that appears the most (multi-view voting).

    (q|g)_sequences: indicates to which sequence the sample belongs to.
    """

    per_person_heatmap = defaultdict(lambda: np.empty((0, distmat.shape[1])))
    
    for q_seq in np.unique(q_sequences):
        # get query indices for the current sequence
        q_idxs = np.where(q_sequences == q_seq)[0]
        q_pid = q_pids[q_idxs]
        pid = q_pid[0]
        assert (q_pid == pid).all(), "All samples in the same sequence should have the same PID"

        # compute cmc curve
        distmat_views = distmat[q_idxs] # (num_query, num_gallery)
        per_person_heatmap[pid] = np.concatenate((per_person_heatmap[pid], distmat_views), axis=0)

    heatmap = np.zeros((len(per_person_heatmap), distmat.shape[1]))
    for i, values in per_person_heatmap.items():
        heatmap[i] = np.mean(values, axis=0)

    output_path = Path("output", "paper", f"heatmap_svm_{modality}_mm_or.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, heatmap)
    print(f"Saved heatmap to {output_path}")
    return heatmap
        