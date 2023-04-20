
from typing import ClassVar
import torch
import scipy

import torch.nn.functional as F
import numpy as np 
import pandas as pd

from torch import nn
from torch.utils.data import Dataset 

from nucl_utils import n2id

class Seq2Tensor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq))
        code = F.one_hot(code, num_classes=5)
        
        code[code[:, 4] == 1] = 0.25
        code = code[:, :4].float()
        return code.transpose(0, 1)

class SeqDataset(Dataset):  
    ds: pd.DataFrame
    size: int
    add_single_channel: bool
    add_reverse_channel: bool
    return_probs: bool
    shift: float | None
    scale: float | None
    
    POINTS: ClassVar[np.ndarray] = np.array([-np.inf, *range(1, 18, 1), np.inf])
    
    def __init__(self, 
                 ds: pd.DataFrame,
                 size: int, 
                 add_single_channel: bool,
                 add_reverse_channel: bool,
                 return_probs: bool,
                 shift: float | None = None, 
                 scale: float | None = None,
                 return_bin: bool=True):
        self.ds = ds
        self.size = size
        self.add_single_channel = add_single_channel
        self.add_reverse_channel = add_reverse_channel
        self.return_probs = return_probs
        self.shift = shift
        self.scale = scale
        if self.return_probs:
            if self.shift is None or self.scale is None:
                raise Exception("To return probs, both shift and scale must be provided")
            if not self.return_bin:
                raise Exception("Return bin must be true if return porbs set to true")
        self.return_bin = return_bin
        self.totensor = Seq2Tensor() 
        
    def transform(self, x):
        assert isinstance(x, str)
        assert len(x) == self.size
        return self.totensor(x)
    
    def bin2prob(self, bin):
        norm = scipy.stats.norm(loc=bin + self.shift, scale=self.scale)
        cumprobs = norm.cdf(self.POINTS)
        probs = cumprobs[1:] - cumprobs[:-1]
        return probs
    
    def __getitem__(self, i):
        seq = self.transform(self.ds.seq.values[i])
        
        to_concat = [seq]
        if self.add_reverse_channel:
            rev = torch.full( (1, self.size), self.ds.rev.values[i], dtype=torch.float32)
            to_concat.append(rev)
        if self.add_single_channel:
            single = torch.full( (1, self.size) , self.ds.is_singleton.values[i], dtype=torch.float32)
            to_concat.append(single)
            
        X = torch.concat(to_concat)
        
        if not self.return_bin:
           return X  
       
        bin = self.ds.bin.values[i] 
        if self.return_probs:
            probs = self.bin2prob(bin)
            return X, probs, bin 
        else:
            return X, bin 
    
    def __len__(self):
        return len(self.ds.seq)

    

