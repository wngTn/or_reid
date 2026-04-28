"""The base model definition.

This module defines the abstract MetaModel class and concrete BaseModel class.
The base model implements data loading, network building, training, and evaluation.

Typical usage:

    BaseModel.run_train(model)
    BaseModel.run_test(model)
"""

import os.path as osp
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
from torch.amp import GradScaler  # type: ignore
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

import data.sampler as Samplers
from data.collate_fn import CollateFn
from data.dataset import DataSet
from data.or_reid_dataset import OrReIDDataset
from data.transform import get_transform
from evaluation import evaluator as eval_functions
from utils import (
    NoOp,
    Odict,
    ddp_all_gather,
    get_attr_from,
    get_msg_mgr,
    get_valid_args,
    is_dict,
    is_list,
    list2var,
    mkdir,
    np2var,
    ts2np,
)
from utils.checkpoints import resolve_checkpoint_hint

from . import backbones
from .loss_aggregator import LossAggregator
from utils.common import get_rank

__all__ = ["BaseModel"]


class MetaModel(metaclass=ABCMeta):
    """The necessary functions for the base model.

    This class defines the necessary functions for the base model, in the base model, we have implemented them.
    """

    @abstractmethod
    def get_loader(self, data_cfg):
        """Based on the given data_cfg, we get the data loader."""
        raise NotImplementedError

    @abstractmethod
    def build_network(self, model_cfg):
        """Build your network here."""
        raise NotImplementedError

    @abstractmethod
    def init_parameters(self):
        """Initialize the parameters of your network."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer(self, optimizer_cfg):
        """Based on the given optimizer_cfg, we get the optimizer."""
        raise NotImplementedError

    @abstractmethod
    def get_scheduler(self, scheduler_cfg):
        """Based on the given scheduler_cfg, we get the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def save_ckpt(self, iteration):
        """Save the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def resume_ckpt(self, restore_hint):
        """Resume the model from the checkpoint, including model parameter, optimizer and scheduler."""
        raise NotImplementedError

    @abstractmethod
    def inputs_pretreament(self, inputs):
        """Transform the input data based on transform setting."""
        raise NotImplementedError

    @abstractmethod
    def train_step(self, loss_num) -> bool:
        """Do one training step."""
        raise NotImplementedError

    @abstractmethod
    def inference(self):
        """Do inference (calculate features.)."""
        raise NotImplementedError

    @abstractmethod
    def run_train(model):
        """Run a whole train schedule."""
        raise NotImplementedError

    @abstractmethod
    def run_test(model):
        """Run a whole test schedule."""
        raise NotImplementedError


class BaseModel(MetaModel, nn.Module):
    """Base model.

    This class inherites the MetaModel class, and implements the basic model functions, like get_loader, build_network, etc.

    Attributes:
        msg_mgr: the massage manager.
        cfgs: the configs.
        iteration: the current iteration of the model.
        engine_cfg: the configs of the engine(train or test).
        save_path: the path to save the checkpoints.

    """

    def __init__(self, cfgs, training, split):
        """Initialize the base model.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """

        super(BaseModel, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.split = split
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs["trainer_cfg"] if training else cfgs["evaluator_cfg"]
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        self.msg_mgr.log_info(cfgs["data_cfg"])
        if training:
            self.train_loader = self.get_loader(cfgs["data_cfg"], train=True)
            cfgs["model_cfg"]["SeparateBNNecks"]["class_num"] = self.train_loader.dataset.class_num
        if not training or self.engine_cfg["with_test"]:
            self.test_loader = self.get_loader(cfgs["data_cfg"], train=False)
            if not training:
                cfgs["model_cfg"]["SeparateBNNecks"]["class_num"] = self.test_loader.dataset.class_num
            self.evaluator_trfs = get_transform(cfgs["evaluator_cfg"]["transform"])

        if training and self.engine_cfg["enable_float16"]:
            self.Scaler = GradScaler()

        self._training_mode = training
        self.build_network(cfgs["model_cfg"])
        self.init_parameters()
        self.trainer_trfs = get_transform(cfgs["trainer_cfg"]["transform"])

        self.device = get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device("cuda", self.device))

        if training:
            self.loss_aggregator = LossAggregator(cfgs["loss_cfg"])
            self.optimizer = self.get_optimizer(self.cfgs["optimizer_cfg"])
            self.scheduler = self.get_scheduler(cfgs["scheduler_cfg"])
        self.train(training)
        restore_hint = self.engine_cfg["restore_hint"]
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        if is_dict(backbone_cfg):
            Backbone = get_attr_from([backbones], backbone_cfg["type"])
            valid_args = get_valid_args(Backbone, backbone_cfg, ["type"])
            return Backbone(**valid_args)
        if is_list(backbone_cfg):
            Backbone = nn.ModuleList([self.get_backbone(cfg) for cfg in backbone_cfg])
            return Backbone
        raise ValueError("Error type for -Backbone-Cfg-, supported: (A list of) dict.")

    def build_network(self, model_cfg):
        if "backbone_cfg" in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg["backbone_cfg"])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize weights for the linear transformations
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                # Initialize biases, if they exist
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0.0)

    def get_loader(self, data_cfg, train=True):
        sampler_cfg = self.cfgs["trainer_cfg"]["sampler"] if train else self.cfgs["evaluator_cfg"]["sampler"]
        if self.cfgs["data_cfg"]["dataset_name"] == "OR_ReID":
            dataset = OrReIDDataset(data_cfg, train, self.split)
        else:
            dataset = DataSet(data_cfg, train)

        Sampler = get_attr_from([Samplers], sampler_cfg["type"])
        vaild_args = get_valid_args(Sampler, sampler_cfg, free_keys=["sample_type", "type"])
        sampler = Sampler(dataset, **vaild_args)

        loader = tordata.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            collate_fn=CollateFn(dataset.label_set, sampler_cfg),
            num_workers=data_cfg["num_workers"],
        )
        return loader

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)

        # Initialize optimizer class
        optimizer_class = get_attr_from([optim], optimizer_cfg["solver"])
        valid_arg = get_valid_args(optimizer_class, optimizer_cfg, ["solver"])

        # Check if lr_dict is provided
        if "lr_dict" in optimizer_cfg:
            lr_dict = optimizer_cfg["lr_dict"]

            # Create parameter groups
            param_groups = []

            # Add groups based on lr_dict
            for group_name, lr in lr_dict.items():
                params = [p for n, p in self.named_parameters() if group_name in n and p.requires_grad]
                param_groups.append({"params": params, "lr": lr})

            # Handle remaining parameters with default lr
            remaining_params = [
                p
                for n, p in self.named_parameters()
                if all(group_name not in n for group_name in lr_dict.keys()) and p.requires_grad
            ]
            if remaining_params:
                param_groups.append({"params": remaining_params, "lr": optimizer_cfg["lr"]})

            # Remove the lr key from valid_arg
            valid_arg.pop("lr", None)
            # Initialize optimizer with parameter groups
            optimizer = optimizer_class(param_groups, **valid_arg)
        else:
            # Initialize optimizer in the standard way if lr_dict is not provided
            optimizer = optimizer_class(filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)

        return optimizer

    def get_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from([optim.lr_scheduler], scheduler_cfg["scheduler"])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ["scheduler"])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        return scheduler

    def save_ckpt(self, iteration):
        if get_rank() == 0:
            mkdir(osp.join(self.msg_mgr.save_path, "checkpoints/"))
            save_name = self.engine_cfg["save_name"]
            checkpoint = {
                "model": self.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "iteration": iteration,
            }
            torch.save(
                checkpoint,
                osp.join(self.msg_mgr.save_path, f"checkpoints/{save_name}_SPLIT_{self.split}-{iteration:0>5}.pt"),
            )

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg["restore_ckpt_strict"]

        checkpoint = torch.load(save_name, map_location=torch.device("cuda", self.device))
        model_state_dict = checkpoint["model"]
        current_state_dict = self.state_dict()

        unexpected = sorted(set(model_state_dict.keys()) - set(current_state_dict.keys()))
        mismatched = sorted(
            key
            for key in set(model_state_dict.keys()) & set(current_state_dict.keys())
            if model_state_dict[key].shape != current_state_dict[key].shape
        )

        if not self.training and unexpected:
            self.msg_mgr.log_warning("-------- Ignored Checkpoint Params (not used during eval) --------")
            self.msg_mgr.log_warning(unexpected)
            model_state_dict = {k: v for k, v in model_state_dict.items() if k in current_state_dict}
            load_ckpt_strict = False

        if mismatched:
            if load_ckpt_strict:
                raise RuntimeError(f"Mismatched checkpoint parameter shapes: {mismatched}")
            self.msg_mgr.log_warning("-------- Ignored Checkpoint Params (shape mismatch) --------")
            self.msg_mgr.log_warning(mismatched)
            model_state_dict = {k: v for k, v in model_state_dict.items() if k not in mismatched}

        if not load_ckpt_strict:
            skipped = sorted(set(model_state_dict.keys()) - set(current_state_dict.keys()))
            if skipped:
                self.msg_mgr.log_info("-------- Skipped Checkpoint Params (not in model) --------")
                self.msg_mgr.log_info(skipped)

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg.get("optimizer_reset", False) and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                self.msg_mgr.log_warning("Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg.get("scheduler_reset", False) and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                self.msg_mgr.log_warning("Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)

    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg["save_name"]
            save_name = osp.join(
                self.msg_mgr.save_path, f"checkpoints/{save_name}_SPLIT_{self.split}-{restore_hint:0>5}.pt"
            )
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = resolve_checkpoint_hint(restore_hint, self.cfgs, self.split)
            self.iteration = 0
        else:
            raise ValueError("Error type for -Restore_Hint-, supported: int or string.")
        self._load_ckpt(save_name)

    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find("BatchNorm") != -1:
                module.eval()

    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(
                    len(seqs_batch), len(seq_trfs)
                )
            )
        requires_grad = bool(self.training)
        seqs = [
            np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
            for trf, seq in zip(seq_trfs, seqs_batch)
        ]

        typs = typs_batch
        vies = vies_batch

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL

    def train_step(self, loss_sum) -> bool:
        """Conduct loss_sum.backward(), self.optimizer.step() and self.scheduler.step().

        Args:
            loss_sum:The loss of the current batch.
        Returns:
            bool: True if the training is finished, False otherwise.
        """

        self.optimizer.zero_grad()
        if loss_sum <= 1e-9:
            self.msg_mgr.log_warning("Find the loss sum less than 1e-9 but the training process will continue!")

        if self.engine_cfg["enable_float16"]:
            self.Scaler.scale(loss_sum).backward()
            if "clip_gradient" in self.engine_cfg and self.engine_cfg["clip_gradient"] > 0.0:
                self.Scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), self.engine_cfg["clip_gradient"])
            self.Scaler.step(self.optimizer)
            scale = self.Scaler.get_scale()
            self.Scaler.update()
            # Warning caused by optimizer skip when NaN
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/5
            if scale != self.Scaler.get_scale():
                self.msg_mgr.log_debug(
                    "Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                        scale, self.Scaler.get_scale()
                    )
                )
                return False
        else:
            loss_sum.backward()
            if "clip_gradient" in self.engine_cfg and self.engine_cfg["clip_gradient"] > 0.0:
                nn.utils.clip_grad_norm_(self.parameters(), self.engine_cfg["clip_gradient"])
            self.optimizer.step()

        self.iteration += 1
        self.scheduler.step()
        return True

    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc="Transforming")
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(device_type="cuda", enabled=self.engine_cfg["enable_float16"]):
                retval = self.forward(ipts)
                inference_feat = retval["inference_feat"]
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                if torch.isnan(v).any():
                    self.msg_mgr.log_warning("Find NaN in the inference feature, please check the model!")
                    continue
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)

            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            info_dict[k] = np.concatenate(v)[:total_size]
        return info_dict

    @staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for inputs in model.train_loader:
            ipts = model.inputs_pretreament(inputs)
            with autocast("cuda", enabled=model.engine_cfg["enable_float16"]):
                retval = model(ipts)
                training_feat = retval["training_feat"]
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                continue

            learning_rates_logged = {}
            for param_group in model.optimizer.param_groups:
                lr = param_group["lr"]
                # Check if this learning rate is already logged
                if lr not in learning_rates_logged:
                    # Find a representative parameter name
                    representative_param = next(iter(param_group["params"]))
                    param_name = next(name for name, p in model.named_parameters() if p is representative_param).split(
                        "."
                    )[1]
                    # Map learning rate to parameter name and log it
                    learning_rates_logged[lr] = param_name

            model.msg_mgr.train_step(loss_info)

            eval_iter = model.engine_cfg.get("eval_iter", 10_000)
            if model.engine_cfg.get("with_test", False) and model.iteration % eval_iter == 0:
                model.msg_mgr.log_info("Running evaluation...")
                model.eval()
                BaseModel.run_test(model, during_train=True)
                model.train()
                if model.cfgs["trainer_cfg"]["fix_BN"]:
                    model.fix_BN()
                model.msg_mgr.reset_time()

            if model.iteration % model.engine_cfg["save_iter"] == 0:
                model.save_ckpt(model.iteration)

            if model.iteration >= model.engine_cfg["total_iter"]:
                break

    @staticmethod
    def run_test(model, during_train=False):
        """Accept the instance object(model) here, and then run the test loop."""

        rank = get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank)
        if rank != 0:
            return {}
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list
            recording_list = loader.dataset.recording_list if hasattr(loader.dataset, "recording_list") else None

            info_dict.update(
                {
                    "labels": label_list,
                    "recordings": recording_list,
                    "types": types_list,
                    "views": views_list,
                    "dataset": loader.dataset,
                }
            )

            if "eval_func" in model.cfgs["evaluator_cfg"].keys():
                eval_func = model.cfgs["evaluator_cfg"]["eval_func"]
            else:
                eval_func = "identification"
            eval_func = getattr(eval_functions, eval_func)
            # Merge all configurations and model into one dictionary
            combined_args = {
                **model.cfgs["evaluator_cfg"],
                **model.cfgs["data_cfg"],
                **{"model": getattr(model, "module", model)},
                **{"transform": model.evaluator_trfs[0]},
                "during_train": during_train,
            }
            free_keys = ["metric", "test_set", "multi_view", "model"]
            valid_args = get_valid_args(eval_func, combined_args, free_keys)
            return eval_func(info_dict, loader.dataset, **valid_args)
