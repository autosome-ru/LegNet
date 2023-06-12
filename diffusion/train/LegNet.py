from torch import nn
from tltorch import TRL
from collections import OrderedDict
import torch


class Bilinear(nn.Module):
    """
    Bilinear layer introduces pairwise product to a NN to model possible combinatorial effects.
    This particular implementation attempts to leverage the number of parameters via low-rank tensor decompositions.
    Parameters
    ----------
    n : int
        Number of input features.
    out : int, optional
        Number of output features. If None, assumed to be equal to the number of input features. The default is None.
    rank : float, optional
        Fraction of maximal to rank to be used in tensor decomposition. The default is 0.05.
    bias : bool, optional
        If True, bias is used. The default is False.
    """
    def __init__(self, n: int, out=None, rank=0.05, bias=False):        
        super().__init__()
        if out is None:
            out = (n, )
        self.trl = TRL((n, n), out, bias=bias, rank=rank)
        self.trl.weight = self.trl.weight.normal_(std=0.00075)
    
    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))

class Concater(nn.Module):
    """
    Concatenates an output of some module with its input alongside some dimension.
    Parameters
    ----------
    module : nn.Module
        Module.
    dim : int, optional
        Dimension to concatenate along. The default is -1.
    """
    def __init__(self, module: nn.Module, dim=-1):        
        super().__init__()
        self.mod = module
        self.dim = dim
    
    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)

class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer.
    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.
    """
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), int(inp // reduction)),
                Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class LegNet(nn.Module):
    """
    NoGINet neural network.
    Parameters
    ----------
    seqsize : int
        Sequence length.
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
                seqsize,
                block_sizes=[256, 256, 128, 128, 64, 64, 32, 32],
                ks=5,
                resize_factor=4,
                activation=nn.SiLU,
                filter_per_group=2,
                se_reduction=4,
                final_ch=18,
                bn_momentum=0.1): 
        super().__init__()
        self.block_sizes = block_sizes
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.seqsize = seqsize
        self.final_ch = final_ch
        self.bn_momentum = bn_momentum
        seqextblocks = OrderedDict()
        
        block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=6,
                            ## CHANGE!!!
                            out_channels=block_sizes[0],
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(block_sizes[0],
                                     momentum=self.bn_momentum),
                       activation()#Exponential(block_sizes[0]) #activation()
        )
        seqextblocks[f'blc0'] = block

        
        for ind, (prev_sz, sz) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            block = nn.Sequential(
                        #nn.Dropout(0.1),
                        nn.Conv1d(
                            in_channels=prev_sz,
                            out_channels=sz * self.resize_factor,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, 
                                      momentum=self.bn_momentum),
                       activation(), #Exponential(sz * self.resize_factor), #activation(),
                       
                       
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=sz * self.resize_factor,
                            kernel_size=ks,
                            groups=sz * self.resize_factor // filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz * self.resize_factor, 
                                      momentum=self.bn_momentum),
                       activation(), #Exponential(sz * self.resize_factor), #activation(),
                       SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                       #nn.Dropout(0.1),
                       nn.Conv1d(
                            in_channels=sz * self.resize_factor,
                            out_channels=prev_sz,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(prev_sz,
                                      momentum=self.bn_momentum),
                       activation(), #Exponential(sz), #activation(),
            
            )
            seqextblocks[f'inv_res_blc{ind}'] = block
            block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=2 * prev_sz,
                            out_channels=sz,
                            kernel_size=ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(sz, 
                                      momentum=self.bn_momentum),
                       activation(),#Exponential(sz), #activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block
        
        self.seqextractor = nn.ModuleDict(seqextblocks)

        self.mapper =  block = nn.Sequential(
                        #nn.Dropout(0.1),
                        nn.Conv1d(
                            in_channels=block_sizes[-1],
                            out_channels=self.final_ch,
                            kernel_size=1,
                            padding='same',
                       ),
#                        activation()
        )
        
        ## REMOVE?
        # self.register_buffer('bins', torch.arange(start=0, end=18, step=1, requires_grad=False))
        
    def feature_extractor(self, x):
        x = self.seqextractor['blc0'](x)
        
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x 

    def forward(self, x):    
        f = self.feature_extractor(x)
        x = self.mapper(f)
        # x = F.adaptive_avg_pool1d(x, 1)
        # x = x.squeeze(2)
        # logprobs = F.log_softmax(x, dim=1) 
        # if predict_score:
        #     x = F.softmax(x, dim=1)
        #     score = (x * self.bins).sum(dim=1)
        #     return logprobs, score
        # else:
        #     return logprobs # modified
        return x