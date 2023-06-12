import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

from typing import Tuple



MUT_POOLS = {0: (3, 1, 2),
             3: (0, 1, 2),
             1: (0, 3, 2),
             2: (0, 3, 1)}

CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

def n2id(n):
    return CODES[n.upper()]

class PromotersData(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 device = torch.device('cuda:1')):
        self.device = device
        self.data = df
        self.seqs = self.data['seq']
        self.score = self.data['expression']
        self.dataframe = df
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_list = [n2id(i) for i in seq]
        code_target = torch.from_numpy(np.array(seq_list))
        code_target = F.one_hot(code_target, num_classes=5)
        code_target[code_target[:, 4] == 1] = 0.25
        seqs_target_encode = (code_target[:, :4].float()).transpose(0, 1)    
        
        out_seq = torch.concat((
                                        seqs_target_encode,
                                        torch.full((1,seqs_target_encode.shape[1]), self.score[index]),
                                        torch.full((1,seqs_target_encode.shape[1]), 0)
                                        ))

        return out_seq