# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F 

from typing import Type
from torch import nn
from residual import Residual, ResidualConcat
from local_block import LocalBlock
from effblock import EffBlock
from mapping_block import MappingBlock
    
class LegNet(nn.Module):
    """
    NoGINet neural network.

    Parameters
    ----------
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.

    """
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                use_single_channel: bool, 
                use_reverse_channel: bool,
                block_sizes: list[int]=[256, 256, 128, 128, 64, 64, 32, 32], 
                ks: int=5, 
                resize_factor: int=4, 
                activation: Type[nn.Module]=nn.SiLU,
                final_activation: Type[nn.Module]=nn.SiLU,
                filter_per_group: int=2,
                se_reduction: int=4,
                res_block_type: str="concat",
                se_type: str="complex",
                inner_dim_calculation: str="out"):        
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.use_single_channel = use_single_channel
        self.use_reverse_channel = use_reverse_channel
        self.filter_per_group = filter_per_group
        self.final_ch = 18 # number of bins in the competition
        self.inner_dim_calculation= inner_dim_calculation
        self.res_block_type = res_block_type
        

        if res_block_type == "concat":
            residual = ResidualConcat
            local_multiplier = 2
        elif res_block_type == "add":
            residual = Residual
            local_multiplier = 1
        elif res_block_type == "none":
            residual = lambda x : x
            local_multiplier = 1
        else:
            raise NotImplementedError()    
        
        self.stem_block = LocalBlock(in_ch=self.in_channels,
                           out_ch=block_sizes[0],
                           ks=ks,
                           activation=activation)

        blocks = []
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                residual(EffBlock(in_ch=prev_sz, 
                         out_ch=sz,
                         ks=ks,
                         resize_factor=4,
                         activation=activation,
                         filter_per_group=self.filter_per_group,
                         se_type=se_type,
                         inner_dim_calculation=inner_dim_calculation)),
                LocalBlock(in_ch=local_multiplier * prev_sz,
                               out_ch=sz,
                               ks=ks,
                               activation=activation)
            )
            blocks.append(block)

        
        self.main = nn.Sequential(*blocks)

        self.mapper =  MappingBlock(in_ch=block_sizes[-1],
                                    out_ch=self.final_ch,
                                    activation=final_activation)
        
        
        self.register_buffer('bins', torch.arange(start=0, end=18, step=1, requires_grad=False))

    @property
    def in_channels(self) -> int:
        return 4 + self.use_reverse_channel + self.use_single_channel
    
    def forward(self, x):    
        x = self.stem_block(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        logprobs = F.log_softmax(x, dim=1) 
        x = F.softmax(x, dim=1)
        score = (x * self.bins).sum(dim=1)
        return logprobs, score
       
