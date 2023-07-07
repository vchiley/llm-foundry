# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""GPT Blocks used for the GPT Model."""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch import distributed

from composer.utils import dist

from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY

try:
    import transformer_engine.pytorch as te
except:
    te = None


class MPTMLP(nn.Module):

    def __init__(
        self,
        d_model: int,
        expansion_ratio: int,
        fc_type: str = 'torch',
        device: Optional[str] = None,
    ):
        super().__init__()
        fc_kwargs = {}
        if fc_type != 'te':
            fc_kwargs['device'] = device
        self.up_proj = FC_CLASS_REGISTRY[fc_type](
            d_model,
            expansion_ratio * d_model,
            **fc_kwargs,
        )
        self.act = nn.GELU(approximate='none')
        self.down_proj = FC_CLASS_REGISTRY[fc_type](
            expansion_ratio * d_model,
            d_model,
            **fc_kwargs,
        )
        self.down_proj._is_residual = True  # type: ignore

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


FFN_CLASS_REGISTRY = {
    'mptmlp': MPTMLP,
}

if te is not None:
    FFN_CLASS_REGISTRY['te_ln_mlp'] = te.LayerNormMLP


def build_ffn(
    d_model: int,
    expansion_ratio: int,
    fc_type: str = 'torch',
    device: Optional[str] = None,
    **kwargs,
):
    ffn_type = kwargs.pop('ffn_type')
    if ffn_type == 'mptmlp':
        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f'MPTMLP got an unexpected keyword argument: {kwargs}')
        return MPTMLP(
            d_model=d_model,
            expansion_ratio=expansion_ratio,
            fc_type=fc_type,
            device=device,
        )
    elif ffn_type == 'te_ln_mlp':
        parallel_mode = kwargs.get('set_parallel_mode', False)
        if parallel_mode:
            if not kwargs.get('sequence_parallel', False):
                warnings.warn(
                    'Unexpected usage: te.LayerNormMLP args are `set_parallel_mode: true` and `sequence_parallel: false`.'
                )
            tp_group = kwargs.get('tp_group', None)
            tp_size = kwargs.get('tp_size', 1)
            if tp_group is None and tp_size == 1:
                warnings.warn(f'tp (sp) not configured correctly and therefore will be disabled.')
                kwargs.pop('set_parallel_mode', None)
                kwargs.pop('sequence_parallel', None)
                kwargs.pop('tp_group', None)
                kwargs.pop('tp_size', None)
            
            if tp_group is None and tp_size != 1:
                world_size = dist.get_world_size()
                if world_size % tp_size != 0:
                    raise RuntimeError(f'{world_size} must be divisible by {tp_size=}.')
                start = dist.get_global_rank() // tp_size * tp_size
                ranks = tuple(range(start, start + tp_size))
                ranks_per_subgroup_list = list(set(dist.all_gather_object(ranks)))
                current_group, _subgroups = distributed.distributed_c10d.new_subgroups_by_enumeration(ranks_per_subgroup_list)
                tp_group = current_group
                kwargs['tp_group'] = tp_group
            
            if tp_group is not None and tp_size == 1:
                tp_size = tp_group.size()
                kwargs['tp_size'] = tp_size

        mlp = te.LayerNormMLP(
            hidden_size=d_model,
            ffn_hidden_size=d_model * expansion_ratio,
            **kwargs,
        )

        if parallel_mode:
            mlp._fsdp_process_group = f"mod{tp_size}"

        return mlp

    raise ValueError(f'{ffn_type=} not recognized.')
