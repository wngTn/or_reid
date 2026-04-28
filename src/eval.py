"""Evaluate OR Re-ID models from checkpoints, with cross-dataset support.

Examples:
    # Evaluate all 4 splits
    python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp

    # Evaluate a single split
    python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp --splits 0

    # Cross-dataset: model trained on OR_ReID_13, tested on 4D-OR_ReID
    python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp --test_dataset 4D-OR_ReID

    # Evaluate a pre-trained checkpoint directly
    python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt sustech1k

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp
"""

import argparse
import os

import pandas as pd
import torch
from modeling import models
from utils import config_loader, get_ddp_module, get_msg_mgr, init_distributed, init_seeds, params_count
from utils.checkpoints import resolve_checkpoint_hint
from utils.common import get_rank
from utils.pprint import log_results
from utils.statistics import aggregate_results

ALL_SPLITS = [0, 1, 2, 3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OR Re-ID models from checkpoints.",
        epilog="""\
examples:
  python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp
  python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp --splits 0
  python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --exp_name my_exp --test_dataset 4D-OR_ReID
  python src/eval.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt sustech1k
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cfgs", type=str, required=True, help="path to config YAML file")
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help="splits to evaluate: 'all' for all 4 (default), or comma-separated indices e.g. '0' or '0,2'",
    )
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name matching the training run")
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        help="override test dataset for cross-dataset evaluation (e.g. '4D-OR_ReID')",
    )
    parser.add_argument("--iter", type=int, default=0, help="checkpoint iteration to evaluate")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "checkpoint file, checkpoint directory, or ckpts/manifest.yaml alias "
            "(e.g. or_reid_13/depth or sustech1k); sets restore_ckpt_strict=false automatically"
        ),
    )
    parser.add_argument("--no_log_file", action="store_true", help="disable writing logs to file")
    return parser.parse_args()


def parse_splits(splits_str):
    """Parse --splits argument into a list of split indices."""
    if splits_str == "all":
        return list(ALL_SPLITS)
    splits = [int(s.strip()) for s in splits_str.split(",")]
    for s in splits:
        if s not in ALL_SPLITS:
            raise ValueError(f"Invalid split {s}. Valid splits: {ALL_SPLITS}")
    return splits


def main():
    opt = parse_args()
    splits = parse_splits(opt.splits)
    cfgs = config_loader(opt.cfgs)
    cfgs["evaluator_cfg"]["restore_ckpt_strict"] = False

    if opt.ckpt:
        cfgs["evaluator_cfg"]["restore_hint"] = opt.ckpt
    elif opt.iter != 0:
        cfgs["evaluator_cfg"]["restore_hint"] = opt.iter

    # Cross-dataset override
    is_cross_dataset = False
    if opt.test_dataset:
        cfgs["data_cfg"]["test_dataset_name"] = opt.test_dataset
        is_cross_dataset = True

    exp_name = opt.exp_name or cfgs["evaluator_cfg"]["save_name"]
    engine_cfg = cfgs["evaluator_cfg"]
    output_path = os.path.join("output", exp_name, engine_cfg["save_name"])

    rank = init_distributed()

    msg_mgr = get_msg_mgr()
    msg_mgr.init_manager(cfgs, output_path, not opt.no_log_file, 0, engine_cfg["restore_hint"], training=False)
    init_seeds(rank, cuda_deterministic=False)

    msg_mgr.log_info(f"Evaluating splits: {splits}")
    if is_cross_dataset:
        msg_mgr.log_info(f"Cross-dataset: train={cfgs['data_cfg']['train_dataset_name']}, test={opt.test_dataset}")
    msg_mgr.log_info(engine_cfg)

    Model = getattr(models, cfgs["model_cfg"]["model"])
    msg_mgr.log_info(cfgs["model_cfg"])
    results = []

    for split in splits:
        if opt.ckpt:
            cfgs["evaluator_cfg"]["restore_hint"] = resolve_checkpoint_hint(opt.ckpt, cfgs, split)
            msg_mgr.log_info(f"Split {split}: using checkpoint {cfgs['evaluator_cfg']['restore_hint']}")
        model = Model(cfgs, False, split)
        model = get_ddp_module(model)

        msg_mgr.log_info(params_count(model))
        msg_mgr.log_info(f"=== Split {split}: Evaluating ===")
        result_dict = Model.run_test(model)

        if result_dict:
            for num_sequence, sequence_dict in result_dict.items():
                for metric, value in sequence_dict.items():
                    results.append({"split": split, "num_sequence": num_sequence, "metric": metric, "value": value})
                msg_mgr.log_info(
                    f"Split {split} | seq={num_sequence} | "
                    f"mAP={sequence_dict.get('mAP', 0):.4f} | "
                    f"CMC={sequence_dict.get('cmc', 0):.4f} | "
                    f"Acc={sequence_dict.get('accuracy', 0):.4f} | "
                    f"Macro={sequence_dict.get('macro_accuracy', 0):.4f}"
                )

        msg_mgr.iteration = 0
        del model
        torch.cuda.empty_cache()

    if get_rank() == 0 and results:
        df = pd.DataFrame(results)
        train_name = cfgs["data_cfg"]["train_dataset_name"]
        test_name = cfgs["data_cfg"]["test_dataset_name"]
        pkl_name = f"results_train_{train_name}_test_{test_name}.pkl" if train_name != test_name else "results.pkl"
        save_path = msg_mgr.save_path / pkl_name
        df.to_pickle(save_path.as_posix())
        msg_mgr.log_info(f"Results saved to {pkl_name}")

        if len(splits) > 1:
            aggregated_df = aggregate_results(df)
            display_metrics = ["mAP", "cmc", "accuracy", "macro_accuracy"]
            display_df = aggregated_df[aggregated_df["metric"].isin(display_metrics)]
            msg_mgr.log_info(log_results(display_df, msg_mgr))


if __name__ == "__main__":
    main()
