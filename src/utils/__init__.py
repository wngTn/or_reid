from .common import MergeCfgsDict as MergeCfgsDict
from .common import NoOp as NoOp
from .common import Ntuple as Ntuple
from .common import Odict as Odict
from .common import clones as clones
from .common import config_loader as config_loader
from .common import ddp_all_gather as ddp_all_gather
from .common import get_attr_from as get_attr_from
from .common import get_ddp_module as get_ddp_module
from .common import get_valid_args as get_valid_args
from .common import init_distributed as init_distributed
from .common import init_seeds as init_seeds
from .common import is_array as is_array
from .common import is_bool as is_bool
from .common import is_dict as is_dict
from .common import is_list as is_list
from .common import is_list_or_tuple as is_list_or_tuple
from .common import is_str as is_str
from .common import is_tensor as is_tensor
from .common import list2var as list2var
from .common import mkdir as mkdir
from .common import np2var as np2var
from .common import params_count as params_count
from .common import ts2np as ts2np
from .common import ts2var as ts2var


def get_msg_mgr():
    from .msg_manager import get_msg_mgr as _get_msg_mgr

    return _get_msg_mgr()
