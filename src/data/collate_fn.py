import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

        # point cloud case
        self.points_in_use = sample_config.get('points_in_use', None)

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = [[] for i in range(feature_num)]
            if len(seqs) == 1:
                valid_frames = ~np.all(seqs[0] == 0, axis=tuple(range(1, seqs[0].ndim)))
                indices = np.where(valid_frames)[0]
                seq_len = len(indices)
            else:
                seq_len = len(seqs[0])
                indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    # Check if we have a point cloud with the right dimensions
                    if seqs[i][j].ndim == 2 and seqs[i][j].shape[1] == 3:
                        # Flatten the indices_map to a 1D array for easier processing
                        indices_map = seqs[-1][j] - 1
                        flattened_indices_map = indices_map.flatten()

                        # Filter the point cloud to include only points that are in the indices_map
                        mask = np.isin(np.arange(seqs[i][j].shape[0]), flattened_indices_map)
                        # Initialize an empty array for the final sampled points
                        sampled_point_indices = np.empty((self.points_in_use['points_num'], 2), dtype='float32')

                        if len(np.where(mask)[0]) >= self.points_in_use['points_num']:
                            # Randomly sample points if there are enough or more points than required
                            sampled_indices = np.random.choice(np.where(mask)[0], self.points_in_use['points_num'], replace=False)
                        else:
                            # Randomly sample points with replacement if there are fewer points than required
                            if mask.sum() != 0:
                                sampled_indices = np.random.choice(np.where(mask)[0], self.points_in_use['points_num'], replace=True)
                            else:
                                sampled_indices = np.random.choice(np.arange(seqs[i][j].shape[0]), self.points_in_use['points_num'], replace=True)

                        # Map indices to 2D coordinates in indices_map and append to each point
                        for idx, point_idx in enumerate(sampled_indices):
                            try:
                                y, x = np.divmod(np.where(flattened_indices_map.astype('int') == point_idx)[0][0], 64)
                            except IndexError:
                                y, x = -1, -1
                            sampled_point_indices[idx] = [x, y]
                        # Update the point cloud in seqs
                        seqs[i][j] = seqs[i][j][sampled_indices]
                        seqs[-1][j] = sampled_point_indices

                    sampled_fras[i].append(seqs[i][j])
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            batch[-1] = np.asarray(seqL_batch)

        batch[0] = fras_batch
        return batch
