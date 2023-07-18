# with replacement!!!
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

from typing import Tuple



MUT_POOLS = {0: (3, 1, 2, 0),
             3: (0, 1, 2, 3),
             1: (0, 3, 2, 1),
             2: (0, 3, 1, 2)}

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
                 cleave_left: int ,
                 cleave_right: int,
                 limits: Tuple[int],
                 device = torch.device):
        self.device = device
        self.limits = limits
        if cleave_left or cleave_right:
            if len(df['seq']) > 80:
                if cleave_right == 0:
                    df['seq'] = df['seq'].str[cleave_left:]
                else:
                    df['seq'] = df['seq'].str[cleave_left:-cleave_right]
            self.data = df
        else:
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
        code_target[code_target[:, 4] == 1] = 0
        seqs_target_encode = (code_target[:, :4].float()).transpose(0, 1)
        
        mut_num = torch.randint(low=self.limits[0], high=self.limits[1], size=(1,))

        mutation_sites = torch.randint(low=0, high=len(seq_list), size=(mut_num,))
        mutation_sites_bool_mask = torch.zeros(len(seq_list)+1)
        for site in mutation_sites:
            mutation_sites_bool_mask[site.to(torch.int64)] = 1 
        
        for site in mutation_sites:
            x = torch.randint(low=0, high=4, size=(1,))
            seq_list[site] = MUT_POOLS[seq_list[site]][x]
            
        mutated_seq = torch.from_numpy(np.array(seq_list))
        mutated_seq = F.one_hot(mutated_seq, num_classes=5)
        mutated_seq[mutated_seq[:, 4] == 1] = 0.25
        mutated_seq = (mutated_seq[:, :4].float()).transpose(0, 1)
        
        seqs_mut_encode = torch.concat((
                                        mutated_seq,
                                        torch.full((1,mutated_seq.shape[1]), self.score[index]),
                                        torch.full((1,mutated_seq.shape[1]), mut_num[0])
                                        ))

        return seqs_target_encode, seqs_mut_encode, mutation_sites_bool_mask
