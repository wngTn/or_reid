import random
from collections import defaultdict

import numpy as np
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metric import cuda_dist, evaluate_many_multi_view
from data.meta import NAMES_2_ID_MAPPING


def _get_dist_matrix(probe_features, gal_features, gal_labels, metric='euc'):
    """Create distance matrix between probe and gallery features."""
    if metric in ['euc', 'cos']:
        dist_matrix = cuda_dist(probe_features, gal_features, metric)

    elif metric in ["lr", "svm", "rf"]:
        gal_features = gal_features.reshape(gal_features.shape[0], -1)
        probe_features = probe_features.reshape(probe_features.shape[0], -1)

        if metric == "lr":
            solver = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=0))
        elif metric == "svm":
            solver = make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf', random_state=0, shrinking=True))
        elif metric == "rf":
            solver = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=0))
        solver = solver.fit(X=gal_features, y=gal_labels)
        predictions = solver.predict_proba(probe_features)
        dist_matrix = predictions

    return dist_matrix


def extended_mm_or_reid(
    data,
    dataset,
    metric='euc',
    num_sequences=[10],
    max_iterations=10,
    probe_gal_set=False,
    during_train=False):

    feature, labels, recordings, types, view = data['embeddings'], data['labels'], data['recordings'], data['types'], data['views']
    labels = np.array(labels)
    view = np.array(view)
    recordings = np.array(recordings)
    types = np.array(types)

    _probe_recordings = dataset.probe_recordings
    _gallery_recordings = dataset.gallery_recordings

    label_features_dict = defaultdict(lambda: defaultdict(list))
    for i, (label, recording, type_) in enumerate(zip(labels, recordings, types)):
        label_features_dict[label][(recording, type_)].append(feature[i])

    result_dict = {
        num_seq: defaultdict(int) for num_seq in num_sequences
    }

    id_accuracy_dict = defaultdict(lambda: defaultdict(list))

    for num_sequence in num_sequences:
        mAP_list = []
        cmc_list = []
        mINP_list = []
        accuracy_list = []

        for _ in tqdm.tqdm(range(max_iterations), desc=f'Processing Num Sequences: {num_sequence}'):
            gal_features, gal_labels, gal_recordings, gal_types = [], [], [], []
            probe_features, probe_labels, probe_recordings, probe_types = [], [], [], []
            for label in sorted(label_features_dict.keys()):
                features_by_key = label_features_dict[label]
                keys = list(features_by_key.keys())

                if probe_gal_set:
                    gallery_keys = [(rec, typ) for rec, typ in keys if rec in _gallery_recordings]
                    probe_keys = [(rec, typ) for rec, typ in keys if rec in _probe_recordings]

                    if len(gallery_keys) > 0:
                        gallery_keys = set(random.sample(gallery_keys, min(len(gallery_keys), num_sequence)))
                    else:
                        continue

                    for (recording, type_), features_list in features_by_key.items():
                        if (recording, type_) in gallery_keys:
                            gal_features.extend(features_list)
                            gal_labels.extend([label] * len(features_list))
                            gal_recordings.extend([recording] * len(features_list))
                            gal_types.extend([type_] * len(features_list))
                        elif (recording, type_) in probe_keys:
                            probe_features.extend(features_list)
                            probe_labels.extend([label] * len(features_list))
                            probe_recordings.extend([recording] * len(features_list))
                            probe_types.extend([type_] * len(features_list))
                else:
                    gallery_keys = set(random.sample(keys, min(len(keys), num_sequence)))

                    for (recording, type_), features_list in features_by_key.items():
                        if (recording, type_) in gallery_keys:
                            gal_features.extend(features_list)
                            gal_labels.extend([label] * len(features_list))
                            gal_recordings.extend([recording] * len(features_list))
                            gal_types.extend([type_] * len(features_list))
                        else:
                            probe_features.extend(features_list)
                            probe_labels.extend([label] * len(features_list))
                            probe_recordings.extend([recording] * len(features_list))
                            probe_types.extend([type_] * len(features_list))

            gal_features = np.array(gal_features)
            gal_labels = np.array(gal_labels)
            gal_recordings = np.array(gal_recordings)
            gal_types = np.array(gal_types)
            probe_features = np.array(probe_features)
            probe_labels = np.array(probe_labels)
            probe_recordings = np.array(probe_recordings)
            probe_types = np.array(probe_types)

            label_dict = NAMES_2_ID_MAPPING[dataset.test_dataset_name]
            int_gal_labels = np.array([label_dict[label] for label in gal_labels], dtype=np.int32)
            int_probe_labels = np.array([label_dict[label] for label in probe_labels], dtype=np.int32)
            int_to_label = {v: k for k, v in label_dict.items()}

            combined = np.stack((probe_labels, probe_recordings, probe_types), axis=1)
            _, probe_sequences = np.unique(combined, axis=0, return_inverse=True)
            dist_matrix = _get_dist_matrix(probe_features, gal_features, int_gal_labels, metric)
            if metric in ["euc", "cos"]:
                cmc, mAP, mINP, accuracy, id_accuracies = evaluate_many_multi_view(dist_matrix, int_probe_labels, int_gal_labels, probe_sequences, metric, max_rank=1)
            else:
                cmc, mAP, mINP, accuracy, id_accuracies = evaluate_many_multi_view(dist_matrix, int_probe_labels, np.sort(np.unique(int_gal_labels)), probe_sequences, metric, max_rank=1)

            mAP_list.append(mAP)
            cmc_list.append(cmc)
            mINP_list.append(mINP)
            accuracy_list.append(accuracy)
            for idx, acc in id_accuracies.items():
                id_accuracy_dict[num_sequence][int_to_label[idx]].append(acc)

        result_dict[num_sequence]['mAP'] = np.mean(mAP_list)
        result_dict[num_sequence]['cmc'] = np.mean(cmc_list)
        result_dict[num_sequence]['mINP'] = np.mean(mINP_list)
        result_dict[num_sequence]['accuracy'] = np.mean(accuracy_list)
        result_dict[num_sequence]['macro_accuracy'] = np.mean([np.mean(acc) for acc in id_accuracy_dict[num_sequence].values()])

        for idx, acc in id_accuracy_dict[num_sequence].items():
            result_dict[num_sequence][f'accuracy_{idx}'] = np.mean(acc)

    return result_dict
