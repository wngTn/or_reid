SPLIT_OVERVIEW = {
    "4D-OR_ReID": {
        0: {
            "train": [
                "recording_holisticortake1",
                "recording_holisticortake3",
                "recording_holisticortake6",
                "recording_holisticortake9",
            ],
            "gallery": ["recording_holisticortake2", "recording_holisticortake5", "recording_holisticortake8"],
            "probe": ["recording_holisticortake4", "recording_holisticortake7", "recording_holisticortake10"],
            "class_num": 5,
        },
        1: {
            "train": [
                "recording_holisticortake4",
                "recording_holisticortake7",
                "recording_holisticortake10",
                "recording_holisticortake6",
            ],
            "gallery": ["recording_holisticortake1", "recording_holisticortake3", "recording_holisticortake9"],
            "probe": ["recording_holisticortake2", "recording_holisticortake5", "recording_holisticortake8"],
            "class_num": 5,
        },
        2: {
            "train": [
                "recording_holisticortake2",
                "recording_holisticortake5",
                "recording_holisticortake8",
                "recording_holisticortake9",
            ],
            "gallery": ["recording_holisticortake4", "recording_holisticortake7", "recording_holisticortake10"],
            "probe": ["recording_holisticortake1", "recording_holisticortake3", "recording_holisticortake6"],
            "class_num": 5,
        },
        3: {
            "train": [
                "recording_holisticortake3",
                "recording_holisticortake5",
                "recording_holisticortake6",
                "recording_holisticortake10",
            ],
            "gallery": ["recording_holisticortake1", "recording_holisticortake4", "recording_holisticortake9"],
            "probe": ["recording_holisticortake2", "recording_holisticortake7", "recording_holisticortake8"],
            "class_num": 5,
        },
    },
    "OR_ReID_13": {
        # Split 0
        0: {
            "train": ["recording_1", "recording_15", "recording_14_1"],
            "gallery": ["recording_3", "recording_4", "recording_12_2"],
            "probe": ["recording_2", "recording_6", "recording_12_1", "recording_12_3", "recording_17"],
            "class_num": 12,
        },
        1: {
            "train": ["recording_1", "recording_4", "recording_6"],
            "gallery": ["recording_17", "recording_2"],
            "probe": [
                "recording_12_2",
                "recording_15",
                "recording_3",
                "recording_12_1",
                "recording_14_1",
                "recording_12_3",
            ],
            "class_num": 11,
        },
        2: {
            "train": ["recording_2", "recording_3", "recording_15"],
            "gallery": ["recording_4", "recording_12_3", "recording_17"],
            "probe": ["recording_12_2", "recording_1", "recording_12_1", "recording_14_1", "recording_6"],
            "class_num": 12,
        },
        3: {
            "train": ["recording_1", "recording_3"],
            "gallery": ["recording_2", "recording_4", "recording_12_1", "recording_12_2", "recording_17"],
            "probe": ["recording_6", "recording_12_3", "recording_14_1", "recording_15"],
            "class_num": 8,
        },
    },
}


NAMES_2_ID_MAPPING = {
    "OR_ReID_13": {str(i): i - 1 for i in range(1, 14)},
    "4D-OR_ReID": {str(i): i - 1 for i in range(1, 6)},
}