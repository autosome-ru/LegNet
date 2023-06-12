import os

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from typing import Tuple



# DNA-DDPM
from sampler import load_table

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
                 limits: Tuple[int] = (1, 10),
                 device = torch.device('cuda:1'),
                 random_state: int = None):
        self.device = device
        self.limits = limits
        self.data = df
        self.seqs = self.data['seq']
        self.score = self.data['expression']
        # self.encode_data()
#        self.encode_data()
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
        
        mut_num = torch.randint(low=self.limits[0], high=self.limits[1], size=(1,))
        
        out_seq = torch.concat((
                                        seqs_target_encode,
                                        torch.full((1,seqs_target_encode.shape[1]), self.score[index]),
                                        torch.full((1,seqs_target_encode.shape[1]), mut_num[0])
                                        ))

        return out_seq


if __name__ == "__main__":
    # DNA-DDPM
    import os

    path = '/home/kekulen/python_scripts/ML/dna-ddpm/DNA-DDPM/data'
    os.chdir(path)
    from sampler import load_table


    PATH_FROM = '/home/kekulen/python_scripts/ML/dna-ddpm/sequences/work/expression_challenge.txt'
    PATH_TO ='/home/kekulen/python_scripts/ML/dna-ddpm/sequences/work/test117.txt'
    df = load_table(PATH_FROM, PATH_TO, numbseqs=1000, start=0, end=17, length=110, N_contain=False, regen=False)
    
    a = PromotersData(df, limits=(9,10))
    a[2]
    a[2]
    dl = DataLoader(a,
                      batch_size=1,
                      num_workers=1,
                      shuffle=True
                     )
    
    # next(iter(dl))
    # seqs_target_encode, seqs_mut_encode, loss_pos_sites, mutation_sites = a[0]
    # seqs_target_encode == seqs_mut_encode
    # torch.full((80,), 1)

    # torch.concat(
    # # F.one_hot(a[0], num_classes=4)
    
    # c = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0,
    #      0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    #      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,
    #      0, 0, 0, 0, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    #      0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    #      0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
    #      0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 1, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    #      1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
    #      0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
    #      1, 0, 1, 1, 0, 0, 1, 0]])
    # d = torch.tensor([[11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11.]])
    # e = torch.tensor([[11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.,
    #     11., 11., 11., 11., 11., 11., 11., 11., 11., 11.]])

    # torch.concat((c,d,e))
    