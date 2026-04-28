import json
import pickle
from pathlib import Path

import numpy as np
import torch.utils.data as tordata
from tqdm import tqdm

from utils import get_msg_mgr
from .meta import SPLIT_OVERVIEW


class OrReIDDataset(tordata.Dataset):
    def __init__(self, data_cfg, training, split):
        """Initialize OR-ReID dataset for a given split.

        Args:
            data_cfg: Data configuration dict with dataset_root, train/test_dataset_name, data_in_use.
            training: Whether to load training or test data.
            split: Cross-validation split index (0-3).
        """
        self.train_person_names = set()
        self.test_person_names = set()
        self.data_overview = self._load_json(Path("./data/all.json"))

        self.train_dataset_name = data_cfg["train_dataset_name"]
        self.test_dataset_name = data_cfg["test_dataset_name"]

        self.train_recordings = SPLIT_OVERVIEW[data_cfg["train_dataset_name"]][split]["train"]
        self.class_num = SPLIT_OVERVIEW[data_cfg["train_dataset_name"]][split]["class_num"]
        self.gallery_recordings = SPLIT_OVERVIEW[data_cfg["test_dataset_name"]][split]["gallery"]
        self.probe_recordings = SPLIT_OVERVIEW[data_cfg["test_dataset_name"]][split]["probe"]

        assert len(set(self.train_recordings) & set(self.gallery_recordings)) == 0
        assert len(set(self.train_recordings) & set(self.probe_recordings)) == 0
        assert len(set(self.gallery_recordings) & set(self.probe_recordings)) == 0

        self._parse_dataset(data_cfg, training)
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.recording_list = [seq_info[1] for seq_info in self.seqs_info]
        self.types_list = [seq_info[2] for seq_info in self.seqs_info]
        self.views_list = [seq_info[3] for seq_info in self.seqs_info]

        self.label_set = sorted(set(self.label_list))
        self.types_set = sorted(set(self.types_list))
        self.views_set = sorted(set(self.views_list))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)

    def __len__(self):
        return len(self.seqs_info)

    def _load_sequence(self, seq_info):
        _, _, _, _, paths = seq_info
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            with open(pth, "rb") as f:
                data = pickle.load(f).astype(np.float32)
                if data[0].ndim == 3 and data[0].shape[2] == 3:
                    data = np.array([np.transpose(array, (2, 0, 1)) for array in data])
            data_list.append(data)
        return data_list

    def __getitem__(self, idx):
        data_list = self._load_sequence(self.seqs_info[idx])
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def _parse_dataset(self, data_cfg, training):
        dataset_name = data_cfg["train_dataset_name"] if training else data_cfg["test_dataset_name"]
        dataset_root = Path(data_cfg["dataset_root"]) / dataset_name

        try:
            data_in_use = data_cfg["data_in_use"]
        except KeyError:
            raise ValueError("Please specify the data_in_use in data_config!")

        train_set = self.train_recordings
        test_set = self.gallery_recordings + self.probe_recordings

        all_recordings = set()
        for person_dir in dataset_root.iterdir():
            if not person_dir.is_dir():
                continue
            for recording_dir in person_dir.iterdir():
                all_recordings.add(recording_dir.name)
                if recording_dir.name in train_set:
                    self.train_person_names.add(person_dir.name)
                elif recording_dir.name in test_set:
                    self.test_person_names.add(person_dir.name)

        train_set = [r for r in train_set if r in all_recordings]
        test_set = [r for r in test_set if r in all_recordings]

        msg_mgr = get_msg_mgr()

        unused_set = all_recordings - set(train_set) - set(test_set)
        if unused_set:
            msg_mgr.log_debug("-------- Unused Recordings --------")
            msg_mgr.log_debug(sorted(unused_set))

        if training:
            msg_mgr.log_info("-------- Train Recording List --------")
        else:
            msg_mgr.log_info("-------- Test Recording List --------")
        recording_list = train_set if training else test_set
        if len(recording_list) >= 3:
            msg_mgr.log_info(f"[{recording_list[0]}, {recording_list[1]}, ..., {recording_list[-1]}]")
        else:
            msg_mgr.log_info(recording_list)

        seqs_info = self._build_seqs_info(dataset_root, data_in_use, training, train_set, test_set)

        msg_mgr.log_info(
            f"-------- Number of Sequences and Samples {'Training' if training else 'Testing'} --------"
        )
        msg_mgr.log_info(
            f"{len({tuple(s[:3]) for s in seqs_info if len(s) >= 3})} sequences, {len(seqs_info)} samples"
        )

        self.seqs_info = seqs_info

    def _build_seqs_info(self, dataset_root, data_in_use, training, train_set, test_set):
        recording_set, person_set = (
            (train_set, self.train_person_names) if training else (test_set, self.test_person_names)
        )

        seqs_info_list = []
        for person in tqdm(person_set, desc="Loading Data"):
            for recording in recording_set:
                recording_path = dataset_root / person / recording
                if not recording_path.exists():
                    continue

                for typ in sorted(d.name for d in recording_path.iterdir() if d.is_dir()):
                    frame_dir = recording_path / typ / "000"
                    frame_pkls = [f.name for f in frame_dir.iterdir() if f.name.startswith("frames")]
                    if not frame_pkls:
                        continue

                    frames = pickle.load(open(frame_dir / frame_pkls[0], "rb"))
                    misc_dict = self.data_overview[person][recording]
                    miscellaneous = self._get_misc_entries(frames, misc_dict)

                    if "sparse" in map(str.lower, miscellaneous) or "super_sparse" in map(str.lower, miscellaneous):
                        continue

                    view_paths = sorted(d.name for d in (recording_path / typ).iterdir() if d.is_dir())
                    seqs_view_info_list = []
                    most_non_zero_views = []

                    for view in view_paths:
                        seq_info = [person, recording, typ, view]
                        seq_path = dataset_root / person / recording / typ / view

                        seq_dirs = sorted(str(d) for d in seq_path.iterdir())
                        seq_dirs = [d for d, use in zip(seq_dirs, data_in_use) if use]

                        if data_in_use[3]:
                            seq_dirs = [d for d in seq_dirs if "empty" not in d]

                            def count_non_zero_frames(path):
                                with open(path, "rb") as f:
                                    data = pickle.load(f)
                                return np.sum(np.all(data == 0, axis=(1, 2, 3))), len(data) // 2

                            seq_dirs_info = [(d, *count_non_zero_frames(d)) for d in seq_dirs]
                            valid_seq_dirs = [
                                d for d, non_zero_count, half_len in seq_dirs_info if non_zero_count <= half_len
                            ]

                            if not valid_seq_dirs and seq_dirs_info:
                                most_non_zero_views.append(
                                    (*seq_info, [d for d, _, _ in seq_dirs_info], [c for _, c, _ in seq_dirs_info])
                                )
                                seq_dirs = []
                            else:
                                seq_dirs = valid_seq_dirs

                        if seq_dirs:
                            seqs_view_info_list.append([*seq_info, seq_dirs])

                    if data_in_use[3] and len(seqs_view_info_list) == 0:
                        most_non_zero_views = sorted(most_non_zero_views, key=lambda x: sum(x[-1]))
                        seqs_view_info_list.append([*most_non_zero_views[0][:-1]])

                    seqs_info_list.extend(seqs_view_info_list)

        return seqs_info_list

    def _get_misc_entries(self, frames, misc_dict):
        misc_values = []
        for entry in misc_dict.values():
            if any(frame in entry["frames"] for frame in frames):
                misc_values.extend(entry["miscellaneous"])
        return misc_values

    @staticmethod
    def _load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
