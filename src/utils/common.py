import copy
import os
import inspect
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import yaml
import random
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict, namedtuple
import omegaconf
import torch.distributed as dist


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs): pass
        return no_op


class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except AttributeError:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList) or isinstance(x, omegaconf.ListConfig)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict) or isinstance(x, omegaconf.DictConfig)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or not isinstance(v, dict):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def ddp_all_gather(features: torch.Tensor, dim: int = 0, requires_grad: bool = True) -> torch.Tensor:
    """
    Gathers `features` across ranks and concatenates along `dim`.

    - If not in distributed mode (or world_size == 1), returns `features` unchanged.
    - If requires_grad=True, keeps local rank's tensor to preserve gradient flow
      (other ranks' tensors are treated as constants).
    """
    if (not dist.is_available()) or (not dist.is_initialized()):
        return features
    world_size = dist.get_world_size()
    if world_size == 1:
        return features

    rank = dist.get_rank()

    # Create output buffers (same shape/device/dtype)
    feature_list = [torch.empty_like(features) for _ in range(world_size)]

    # all_gather requires contiguous
    dist.all_gather(feature_list, features.contiguous())

    # Preserve autograd path for local rank if requested
    if requires_grad:
        feature_list[rank] = features

    return torch.cat(feature_list, dim=dim)


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_ddp_module(module, **kwargs):
    # loss modules etc.
    if len(list(module.parameters())) == 0:
        return module

    # Not in distributed mode -> return as-is
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_world_size() == 1:
        return module

    device = torch.cuda.current_device()
    module = DDPPassthrough(
        module,
        device_ids=[device],
        output_device=device,
        find_unused_parameters=False,
        **kwargs,
    )
    return module


def params_count(net):
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    
    return f'Total Parameters: {total_params:.5f}M, Trainable Parameters: {trainable_params:.5f}M'

def get_rank():
    if not torch.distributed.is_available():  # type: ignore
        return 0
    if not torch.distributed.is_initialized():  # type: ignore
        return 0
    return torch.distributed.get_rank()  # type: ignore

def get_world_size():
    if not torch.distributed.is_available():  # type: ignore
        return 1
    if not torch.distributed.is_initialized():  # type: ignore
        return 1
    return torch.distributed.get_world_size()  # type: ignore


def init_distributed():
    """Initialize distributed training. Works with and without torchrun.

    When launched via torchrun, reads RANK/LOCAL_RANK/WORLD_SIZE from env
    and initializes NCCL process group. For single-GPU, this is a no-op.

    Returns:
        int: The global rank (0 for single-GPU).
    """
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank
