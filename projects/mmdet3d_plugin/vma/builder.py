import torch.nn as nn
from mmcv.utils import Registry, build_from_cfg

ATTR_HEAD = Registry('Attr_Head')

def build_attr_head(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, ATTR_HEAD, default_args)
