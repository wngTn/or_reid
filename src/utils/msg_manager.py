import logging
import os.path as osp
import time
from pathlib import Path
from time import localtime, strftime


import numpy as np

from .common import NoOp, Odict, get_rank, is_list, is_tensor, list2var, mkdir, ts2np


class MessageManager:
    def __init__(self):
        self.info_dict = Odict()
        self.writer_hparams = ["image", "scalar", "table", "figure", "matrix", "plot"]
        self.time = time.time()

    def init_manager(self, cfgs, save_path, log_to_file, log_iter, iteration=0, training=True):
        self.cfgs = cfgs
        self.iteration = iteration
        self.log_iter = log_iter
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.training = training
        self.init_logger(save_path, log_to_file)

    def init_logger(self, save_path, log_to_file):
        # init logger
        self.logger = logging.getLogger("orion")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        if log_to_file:
            mkdir(osp.join(save_path, "logs/"))
            if self.training:
                log_path = strftime("%Y-%m-%d-%H-%M-%S", localtime()) + ".txt"
            else:
                log_path = f"train_{self.cfgs['data_cfg']['train_dataset_name']}_test_{self.cfgs['data_cfg']['test_dataset_name']}" + strftime("%Y-%m-%d-%H-%M-%S", localtime()) + ".txt"
            vlog = logging.FileHandler(
                osp.join(
                    save_path,
                    "logs/",
                    log_path
                )
            )
            vlog.setLevel(logging.INFO)
            vlog.setFormatter(formatter)
            self.logger.addHandler(vlog)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        self.logger.addHandler(console)

    def append(self, info):
        for k, v in info.items():
            v = [v] if not is_list(v) else v
            v = [ts2np(_) if is_tensor(_) else _ for _ in v]
            info[k] = v
        self.info_dict.append(info)

    def flush(self):
        self.info_dict.clear()

    def write_to_wandb(self, summary):
        import wandb

        # Prepare the data for logging
        log_data = {}
        for k, v in summary.items():
            module_name = k.split("/")[0]
            if module_name not in self.writer_hparams:
                self.log_warning(
                    "Not Expected --Summary-- type [{}] appear!!!{}".format(
                        k, self.writer_hparams
                    )
                )
                continue
            board_name = k.replace(module_name + "/", "")
            if is_list(v):
                v = list2var(v)
            if is_tensor(v):
                v = v.detach()

            if module_name == "scalar":
                try:
                    v = v.mean()
                except AttributeError:
                    pass
            log_data[board_name] = v

        # Log the data to wandb
        wandb.log(log_data, step=self.iteration)

    def log_training_info(self):
        now = time.time()
        string = "Iteration {:0>5}, Cost {:.2f}s".format(
            self.iteration, now - self.time
        )
        for i, (k, v) in enumerate(self.info_dict.items()):
            if "scalar" not in k:
                continue
            k = k.replace("scalar/", "").replace("/", "_")
            end = "\n" if i == len(self.info_dict) - 1 else ""
            string += ", {0}={1:.4f}".format(k, np.mean(v))
            string += end
        self.log_info(string)
        self.reset_time()

    def reset_time(self):
        self.time = time.time()

    def train_step(self, info):
        self.iteration += 1
        self.append(info)
        if self.iteration % self.log_iter == 0:
            try:
                self.log_training_info()
            except Exception as e:
                self.log_warning(e)
            self.flush()
            self.write_to_wandb(info)

    def log_debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def log_info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def log_warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)


msg_mgr = MessageManager()
noop = NoOp()


def get_msg_mgr():
    if get_rank() > 0:
        return noop
    return msg_mgr
