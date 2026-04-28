"""Train OR Re-ID models with configurable cross-validation splits.

Examples:
    # Full 4-fold cross-validation (default)
    python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml

    # Train a single split
    python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --splits 0

    # Train specific splits
    python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --splits 0,2

    # Fine-tune from a pre-trained checkpoint group
    python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt sustech1k --splits 0

    # Multi-GPU via torchrun
    torchrun --nproc_per_node=4 src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn

from modeling import models
from utils import config_loader, get_ddp_module, get_msg_mgr, init_distributed, init_seeds, params_count
from utils.common import get_rank
from utils.pprint import generate_latex_row, log_results
from utils.statistics import aggregate_results

ALL_SPLITS = [0, 1, 2, 3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OR Re-ID models.",
        epilog="""\
examples:
  python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml
  python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --splits 0
  python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --splits 0,2
  python src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml --ckpt sustech1k --splits 0
  torchrun --nproc_per_node=4 src/train.py --cfgs configs/PointCloud/intra/OR_ReID_13.yaml
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cfgs", type=str, required=True, help="path to config YAML file")
    parser.add_argument(
        "--splits",
        type=str,
        default="all",
        help="splits to train: 'all' for full 4-fold CV (default), or comma-separated indices e.g. '0' or '0,2'",
    )
    parser.add_argument("--exp_name", type=str, default=None, help="experiment name (default: config save_name)")
    parser.add_argument("--iter", type=int, default=0, help="iteration to restore from checkpoint")
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
    import wandb

    splits = parse_splits(opt.splits)
    cfgs = config_loader(opt.cfgs)

    if opt.ckpt:
        cfgs["trainer_cfg"]["restore_hint"] = opt.ckpt
        cfgs["trainer_cfg"]["restore_ckpt_strict"] = False
    elif opt.iter != 0:
        cfgs["trainer_cfg"]["restore_hint"] = opt.iter

    exp_name = opt.exp_name or cfgs["trainer_cfg"]["save_name"]
    engine_cfg = cfgs["trainer_cfg"]
    output_path = os.path.join("output", exp_name, engine_cfg["save_name"])

    rank = init_distributed()

    msg_mgr = get_msg_mgr()
    msg_mgr.init_manager(
        cfgs,
        output_path,
        not opt.no_log_file,
        engine_cfg["log_iter"],
        engine_cfg["restore_hint"] if isinstance(engine_cfg["restore_hint"], int) else 0,
    )
    init_seeds(rank, cuda_deterministic=False)

    wandb.init(project=opt.cfgs.split("/")[1], config=cfgs, dir=output_path, mode=os.environ.get("WANDB_MODE", "offline"))

    msg_mgr.log_info(f"Training splits: {splits}")
    msg_mgr.log_info(engine_cfg)

    Model = getattr(models, cfgs["model_cfg"]["model"])
    msg_mgr.log_info(cfgs["model_cfg"])
    results = []

    for split in splits:
        model = Model(cfgs, True, split)
        if engine_cfg["sync_BN"]:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if engine_cfg["fix_BN"]:
            model.fix_BN()
        model = get_ddp_module(model)

        msg_mgr.log_info(params_count(model))
        msg_mgr.log_info(f"=== Split {split}: Training ===")
        Model.run_train(model)

        result_dict = {}
        if engine_cfg["with_test"]:
            msg_mgr.log_info(f"=== Split {split}: Testing ===")
            result_dict = Model.run_test(model)

        if result_dict:
            for num_sequence, sequence_dict in result_dict.items():
                for metric, value in sequence_dict.items():
                    results.append({"split": split, "num_sequence": num_sequence, "metric": metric, "value": value})
            msg_mgr.log_info(f"Split {split} results: {result_dict}")

        msg_mgr.iteration = 0
        del model
        torch.cuda.empty_cache()

    if get_rank() == 0 and results:
        df = pd.DataFrame(results)
        df.to_pickle((msg_mgr.save_path / "results.pkl").as_posix())
        msg_mgr.log_info("Results saved to results.pkl")

        if len(splits) > 1:
            aggregated_df = aggregate_results(df)
            msg_mgr.log_info(log_results(aggregated_df, msg_mgr))
            msg_mgr.log_info(generate_latex_row(aggregated_df, ["mAP", "cmc", "accuracy", "macro_accuracy"]))


if __name__ == "__main__":
    main()
