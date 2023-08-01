# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F 
import random 
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from torch.utils.data import Sampler
from torch import Tensor 
from typing import Sequence, Iterator, ClassVar
import scipy
import json

from model import SeqNN


LEFT_ADAPTER = "TGCATTTTTTTCACATC" 
RIGHT_ADAPTER = "GGTTACGGCTGTT"

PLASMID = "aactctcaaggatcttaccgctgttgagatccagttcgatgtaacccactcgtgcacccaactgatcttcagcatcttttactttcaccagcgtttctgggtgagcaaaaacaggaaggcaaaatgccgcaaaaaagggaataagggcgacacggaaatgttgaatactcatactcttcctttttcaatattattgaagcatttatcagggttattgtctcatgagcggatacatatttgaatgtatttagaaaaataaacaaataggggttccgcgcacatttccccgaaaagtgccacctgacgtcatctatattaccctgttatccctagcggatctgccggtagaggtgtggtcaataagagcgacctcatactatacctgagaaagcaacctgacctacaggaaagagttactcaagaataagaattttcgttttaaaacctaagagtcactttaaaatttgtatacacttattttttttataacttatttaataataaaaatcataaatcataagaaattcgcttatttagaagtGGCGCGCCGGTCCGttacttgtacagctcgtccatgccgccggtggagtggcggccctcggcgcgttcgtactgttccacgatggtgtagtcctcgttgtgggaggtgatgtccaacttgatgttgacgttgtaggcgccgggcagctgcacgggcttcttggccttgtaggtggtcttgacctcagcgtcgtagtggccgccgtccttcagcttcagcctctgcttgatctcgcccttcagggcgccgtcctcggggtacatccgctcggaggaggcctcccagcccatggtcttcttctgcattacggggccgtcggaggggaagttggtgccgcgcagcttcaccttgtagatgaactcgccgtcctgcagggaggagtcctgggtcacggtcaccacgccgccgtcctcgaagttcatcacgcgctcccacttgaagccctcggggaaggacagcttcaagtagtcggggatgtcggcggggtgcttcacgtaggccttggagccgtacatgaactgaggggacaggatgtcccaggcgaagggcagggggccacccttggtcaccttcagcttggcggtctgggtgccctcgtaggggcggccctcgccctcgccctcgatctcgaactcgtggccgttcacggagccctccatgtgcaccttgaagcgcatgaactccttgatgatggccatgttatcctcctcgcccttgctcacCATGGTACTAGTGTTTAGTTAATTATAGTTCGTTGACCGTATATTCTAAAAACAAGTACTCCTTAAAAAAAAACCTTGAAGGGAATAAACAAGTAGAATAGATAGAGAGAAAAATAGAAAATGCAAGAGAATTTATATATTAGAAAGAGAGAAAGAAAAATGGAAAAAAAAAAATAGGAAAAGCCAGAAATAGCACTAGAAGGAGCGACACCAGAAAAGAAGGTGATGGAACCAATTTAGCTATATATAGTTAACTACCGGCTCGATCATCTCTGCCTCCAGCATAGTCGAAGAAGAATTTTTTTTTTCTTGAGGCTTCTGTCAGCAACTCGTATTTTTTCTTTCTTTTTTGGTGAGCCTAAAAAGTTCCCACGTTCTCTTGTACGACGCCGTCACAAACAACCTTATGGGTAATTTGTCGCGGTCTGGGTGTATAAATGTGTGGGTGCAACATGAATGTACGGAGGTAGTTTGCTGATTGGCGGTCTATAGATACCTTGGTTATGGCGCCCTCACAGCCGGCAGGGGAAGCGCCTACGCTTGACATCTACTATATGTAAGTATACGGCCCCATATATAggccctttcgtctcgcgcgtttcggtgatgacggtgaaaacctctgacacatgcagctcccggagacggtcacagcttgtctgtaagcggatgccgggagcagacaagcccgtcagggcgcgtcagcgggtgttggcgggtgtcggggctggcttaactatgcggcatcagagcagattgtactgagagtgcaccatatggacatattgtcgttagaacgcggctacaattaatacataaccttatgtatcatacacatacgatttaggtgacactatagaacgcggccgccagctgaagctttaactatgcggcatcagagcagattgtactgagagtgcaccataccaccttttcaattcatcattttttttttattcttttttttgatttcggtttccttgaaatttttttgattcggtaatctccgaacagaaggaagaacgaaggaaggagcacagacttagattggtatatatacgcatatgtagtgttgaagaaacatgaaattgcccagtattcttaacccaactgcacagaacaaaaacctgcaggaaacgaagataaatcatgtcgaaagctacatataaggaacgtgctgctactcatcctagtcctgttgctgccaagctatttaatatcatgcacgaaaagcaaacaaacttgtgtgcttcattggatgttcgtaccaccaaggaattactggagttagttgaagcattaggtcccaaaatttgtttactaaaaacacatgtggatatcttgactgatttttccatggagggcacagttaagccgctaaaggcattatccgccaagtacaattttttactcttcgaagacagaaaatttgctgacattggtaatacagtcaaattgcagtactctgcgggtgtatacagaatagcagaatgggcagacattacgaatgcacacggtgtggtgggcccaggtattgttagcggtttgaagcaggcggcagaagaagtaacaaaggaacctagaggccttttgatgttagcagaattgtcatgcaagggctccctatctactggagaatatactaagggtactgttgacattgcgaagagcgacaaagattttgttatcggctttattgctcaaagagacatgggtggaagagatgaaggttacgattggttgattatgacacccggtgtgggtttagatgacaagggagacgcattgggtcaacagtatagaaccgtggatgatgtggtctctacaggatctgacattattattgttggaagaggactatttgcaaagggaagggatgctaaggtagagggtgaacgttacagaaaagcaggctgggaagcatatttgagaagatgcggccagcaaaactaaaaaactgtattataagtaaatgcatgtatactaaactcacaaattagagcttcaatttaattatatcagttattaccctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagtGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGAAGGCAAAGatgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctgaccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagagaccacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaataaggcgcgccacttctaaataagcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaattcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccggcagatccgctagggataacagggtaatataGATCTGTTTAGCTTGCCTCGTCCCCGCCGGGTCACCCGGCCAGCGACATGGAGGCCCAGAATACCCTCCTTGACAGTCTTGACGTGCGCAGCTCAGGGGCATGATGTGACTGTCGCCCGTACATTTAGCCCATACATCCCCATGTATAATCATTTGCATCCATACATTTTGATGGCCGCACGGCGCGAAGCAAAAATTACGGCTCCTCGCTGCAGACCTGCGAGCAGGGAAACGCTCCCCTCACAGACGCGTTGAATTGTCCCCACGCCGCGCCCCTGTAGAGAAATATAAAAGGTTAGGATTTGCCACTGAGGTTCTTCTTTCATATACTTCCTTTTAAAATCTTGCTAGGATACAGTTCTCACATCACATCCGAACATAAACAACCATGGGTACCACTCTTGACGACACGGCTTACCGGTACCGCACCAGTGTCCCGGGGGACGCCGAGGCCATCGAGGCACTGGATGGGTCCTTCACCACCGACACCGTCTTCCGCGTCACCGCCACCGGGGACGGCTTCACCCTGCGGGAGGTGCCGGTGGACCCGCCCCTGACCAAGGTGTTCCCCGACGACGAATCGGACGACGAATCGGACGACGGGGAGGACGGCGACCCGGACTCCCGGACGTTCGTCGCGTACGGGGACGACGGCGACCTGGCGGGCTTCGTGGTCGTCTCGTACTCCGGCTGGAACCGCCGGCTGACCGTCGAGGACATCGAGGTCGCCCCGGAGCACCGGGGGCACGGGGTCGGGCGCGCGTTGATGGGGCTCGCGACGGAGTTCGCCCGCGAGCGGGGCGCCGGGCACCTCTGGCTGGAGGTCACCAACGTCAACGCACCGGCGATCCACGCGTACCGGCGGATGGGGTTCACCCTCTGCGGCCTGGACACCGCCCTGTACGACGGCACCGCCTCGGACGGCGAGCAGGCGCTCTACATGAGCATGCCCTGCCCCTAATCAGTACTGACAATAAAAAGATTCTTGTTTTCAAGAACTTGTCATTTGTATAGTTTTTTTATATTGTAGTTGTTCTATTTTAATCAAATGTTAGCGTGATTTATATTTTTTTTCGCCTCGACATCATCTGCCCAGATGCGAAGTTAAGTGCGCAGAAAGTAATATCATGCGTCAATCGTATGTGAATGCTGGTCGCTATACTGCTGTCGATTCGATACTAACGCCGCCATCCAGTGTCGAAAACGAGCTCGaattcctgggtccttttcatcacgtgctataaaaataattataatttaaattttttaatataaatatataaattaaaaatagaaagtaaaaaaagaaattaaagaaaaaatagtttttgttttccgaagatgtaaaagactctagggggatcgccaacaaatactaccttttatcttgctcttcctgctctcaggtattaatgccgaattgtttcatcttgtctgtgtagaagaccacacacgaaaatcctgtgattttacattttacttatcgttaatcgaatgtatatctatttaatctgcttttcttgtctaataaatatatatgtaaagtacgctttttgttgaaattttttaaacctttgtttatttttttttcttcattccgtaactcttctaccttctttatttactttctaaaatccaaatacaaaacataaaaataaataaacacagagtaaattcccaaattattccatcattaaaagatacgaggcgcgtgtaagttacaggcaagcgatccgtccGATATCatcagatccactagtggcctatgcggccgcggatctgccggtctccctatagtgagtcgtattaatttcgataagccaggttaacctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcaTAgctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaagAacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgTtcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtggtcctgcaactttatccgcctccatccagtctattaattgttgccgggaagctagagtaagtagttcgccagttaatagtttgcgcaacgttgttgccattgctacaggcatcgtggtgtcacgctcgtcgtttggtatggcttcattcagctccggttcccaacgatcaaggcgagttacatgatcccccatgttgtgcaaaaaagcggttagctccttcggtcctccgatcgttgtcagaagtaagttggccgcagtgttatcactcatggttatggcagcactgcataattctcttactgtcatgccatccgtaagatgcttttctgtgactggtgagtactcaaccaagtcattctgagaatagtgtatgcggcgaccgagttgctcttgcccggcgtcaatacgggataataccgcgccacatagcagaactttaaaagtgctcatcattggaaaacgttcttcggggcgaa"
PLASMID = PLASMID.upper()
INSERT_START = PLASMID.find('N'*80)

def preprocess_data(data, length):
    data = data.copy()
    add_part = PLASMID[INSERT_START-length:INSERT_START]
    data.seq = data.seq.apply(lambda x:  add_part + x[len(LEFT_ADAPTER):])
    data.seq = data.seq.str.slice(-length, None)
    return data

def set_global_seed(seed: int) -> None:
    """
    Sets random seed into PyTorch, TensorFlow, Numpy and Random.
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True #type: ignore
    torch.backends.cudnn.benchmark = False #type: ignore

CODES = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

INV_CODES = {value: key for key, value in CODES.items()}

COMPL = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'N': 'N'
}

def n2id(n):
    return CODES[n.upper()]

def id2n(i):
    return INV_CODES[i]

def n2compl(n):
    return COMPL[n.upper()]

class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str) -> torch.Tensor:
        seq_i = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq_i))
        code = F.one_hot(code, num_classes=5) # 5th class is N

        code = code[:, :5].float()
        code[code[:, 4] == 1] = 0.25 # encode Ns with .25
        code =  code[:, :4]
        return code.transpose(0, 1)

def parameter_count(model):
    pars = 0  
    for _, p  in model.named_parameters():    
        pars += torch.prod(torch.tensor(p.shape))
    return pars

class SeqDatasetRev(Dataset):
    def __init__(self, ds_rev, size, use_single_channel):
        self.ds = ds_rev
        self.size = size
        self.use_single_channel = use_single_channel
        self.totensor = Seq2Tensor() 
        
    def transform(self, x):
        assert isinstance(x, str)
        assert len(x) == self.size
        return self.totensor(x)
    
    def __getitem__(self, i):
        seq = self.transform(self.ds.seq.values[i])
        rev = torch.full( (1, self.size), self.ds.rev.values[i], dtype=torch.float32)
        
        if self.use_single_channel:
            single = torch.full( (1, self.size) , self.ds.is_singleton.values[i], dtype=torch.float32)
            X = torch.concat([seq, rev, single])
        else:
            X = torch.concat([seq, rev], dim=0)
        
        bin = self.ds.bin.values[i]
        return X, bin 
    
    def __len__(self):
        return len(self.ds.seq)
    
POINTS = np.array([-np.inf, *range(1, 18, 1), np.inf])
class SeqDatasetRevProb(Dataset):
    def __init__(self, ds_rev, size, use_single_channel, shift=0.5, scale=1.5):
        self.ds = ds_rev
        self.size = size
        self.totensor = Seq2Tensor() 
        try:
            self.scale = float(scale)
        except ValueError:
            self.scale = scale
            print("Using adaptive scale")
        self.shift = shift 
        self.use_single_channel = use_single_channel
        
    def transform(self, x):
        assert isinstance(x, str)
        assert len(x) == self.size
        return self.totensor(x)
    
    def __getitem__(self, i):
        seq = self.transform(self.ds.seq.values[i])
        rev = torch.full( (1, self.size), self.ds.rev.values[i], dtype=torch.float32)
        if self.use_single_channel:
            single = torch.full( (1, self.size) , self.ds.is_singleton.values[i], dtype=torch.float32)
            X = torch.concat([seq, rev, single], dim=0)
        else:
            X = torch.concat([seq, rev], dim=0)
        bin = self.ds.bin.values[i]
        if isinstance(self.scale, float):
            norm = scipy.stats.norm(loc=bin + self.shift, scale=self.scale)
        elif self.scale == "adaptive":
            s = -0.1364 * bin + 2.7727
            norm = scipy.stats.norm(loc=bin + self.shift, scale=s)
        else:
            raise Exception("Wrong scale")
        
        cumprobs = norm.cdf(POINTS)
        probs = cumprobs[1:] - cumprobs[:-1]
        return X, probs, bin
    
    def __len__(self):
        return len(self.ds.seq)

class CustomWeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).
    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """
    weights: Tensor
    num_samples: int
    replacement: bool

    SAMPLES_PER_GROUP: ClassVar[int] = 2 ** 12

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))    
        
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.groups, self.group_weights, self.inner_weights = self.groups_split()

    def __iter__(self) -> Iterator[int]:
        group_ids = torch.multinomial(self.group_weights, self.num_samples, replacement=True, generator=self.generator)
        gr, counts = torch.unique(group_ids , sorted=True, return_counts=True)
        smpls = []
        for i in range(len(gr)):
            g = gr[i]
            rand_tensor = torch.multinomial(self.inner_weights[g], counts[i], self.replacement, generator=self.generator)
            idxs = self.groups[g][0] + rand_tensor
            smpls.append(idxs)
        rand_tensor = torch.concat(smpls)
        pi = torch.randperm(rand_tensor.shape[0])
        rand_tensor = rand_tensor[pi] 
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def groups_split(self):
        groups = []
        N = len(self.weights)
        for start in range(0, N, self.SAMPLES_PER_GROUP):
            end = min(start+self.SAMPLES_PER_GROUP, N)
            groups.append(torch.arange(start, end, 1, dtype=torch.long))
        inner_weights = [self.weights[ids] for ids in groups]
        weights = torch.FloatTensor([w.sum() for w in inner_weights])
        return groups, weights, inner_weights

def get_weights(df, tp):
    if tp == "uniform":
        weights = np.full_like(df.bin.values, fill_value=1)
        return weights
    elif tp == "counts":
        weights = df.cnt.values
        return weights
    raise NotImplementedError()


class DataloaderWrapper:
    def __init__(self, dataloader, batch_per_epoch):
        self.batch_per_epoch = batch_per_epoch
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __len__(self):
        return self.batch_per_epoch
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)

    def __iter__(self):
        for _ in range(self.batch_per_epoch):
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)

def revcomp(seq):
    return "".join((n2compl(x) for x in reversed(seq)))


def get_rev(df):
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    revdf['rev'] = 1
    return revdf
    

def add_rev(df):
    df = df.copy()
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    df['rev'] = 0
    revdf['rev'] = 1
    df = pd.concat([df, revdf]).reset_index(drop=True)
    return df

from collections import Counter

def infer_singleton(arr, method):
    if method == "integer":
        return np.array([x.is_integer() for x in arr])
    elif method.startswith("threshold"):
        th = float(method.replace("threshold", ""))
        cnt = Counter(arr)
        return np.array([cnt[x] >= th for x in arr])
    else:
        raise Exception("Wrong method")


def add_singleton_column(df, method):
    df = df.copy()
    df["is_singleton"] = infer_singleton(df.bin.values,method)
    return df 
            

from argparse import ArgumentParser

parser = ArgumentParser()
TRAIN_VAL_PATH = "expression_challenge.txt"

parser.add_argument("--valid_folds", nargs='+',  type=int, default=list())
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--valid_batch_size", type=int, default=4098)
parser.add_argument("--valid_workers", type=int, default=8)
parser.add_argument("--batch_per_epoch", type=int, default=1000)
parser.add_argument("--seqsize", type=int, default=120)
parser.add_argument("--temp", default=".TEMPDIR", type=Path)
parser.add_argument("--use_single_channel", action="store_true")
parser.add_argument("--singleton_definition", choices=["integer", "threshold1100"], default="integer")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--ks", default=5, type=int, help="kernel size of convolutional layers")
parser.add_argument("--blocks", default=[256, 256, 128, 128, 64, 64, 32, 32], nargs="+", type=int, help="number of channels for EffNet-like blocks")
parser.add_argument("--resize_factor", default=4, type=int, help="number of channels in a middle/high-dimensional convolutional layer of an EffNet-like block")
parser.add_argument("--se_reduction", default=4, type=float, help="reduction number used in SELayer")
parser.add_argument("--loss", choices=["mse", "kl"], default="mse", type=str, help="loss function")
parser.add_argument("--final_ch", default=18, type=int, help="number of channels of the final convolutional layer")
parser.add_argument("--target", type=str, help="path to a testing dataset")
parser.add_argument("--model", type=str, default="model/model_80.pth", help="path to a .pth file where parameters are stored")
parser.add_argument("--output", type=str, default="results.tsv", help="path to file where results will be stored")
parser.add_argument("--output_format", type=str, choices=["tsv", "csv", "json"], default="tsv", help="foramt of the output file")
parser.add_argument("--delimiter", default='space', type=str, help="delimiter that separates columns in a training file")


args = parser.parse_args()

print("Loading model...")
device = torch.device(f"cuda:{args.gpu}")
print(device)
model = SeqNN(seqsize=args.seqsize, use_single_channel=args.use_single_channel, block_sizes= args.blocks, ks=args.ks, 
              resize_factor=args.resize_factor, se_reduction=args.se_reduction, final_ch=args.final_ch).to(device)
model.load_state_dict(torch.load(args.model, map_location=device))
model.eval()



print('Parameters:', int(parameter_count(model)))


print(f"Reading dataset {args.target}...")
folds = args.valid_folds
if folds:
    cols = ['seq', 'bin']
else:
    cols = ['seq', 'bin', 'fold']

#df = pd.read_csv(args.target, sep='\t', header=None, names=['seq', 'bin', 'fold'])
df = pd.read_table(args.target, 
     sep='\t' if args.delimiter == 'tab' else ' ', 
     header=None) # modified
if len(df.columns) == 1:
    print(f'Warning: there is just one column in the dataframe. Are you sure that you provided the correct delimiter? Using {args.delimiter} now.')
df.columns = ['seq', 'bin', 'fold'][:len(df.columns)]
print(df.head())
df_orig = df.copy()
df = preprocess_data(df, args.seqsize)
if args.use_single_channel:
    df = add_singleton_column(df, args.singleton_definition)
if folds:
    df = df[df.fold.isin(folds)]
#df = add_rev(df)
df_rev = get_rev(df)
df['rev'] = 0
#tmp_filename = args.target + '.tmp'
ds = SeqDatasetRev(df, size=args.seqsize, use_single_channel=args.use_single_channel)
ds_rev = SeqDatasetRev(df_rev, size=args.seqsize, use_single_channel=args.use_single_channel)
dl = DataLoader(ds, 
                batch_size=args.valid_batch_size, 
                num_workers=args.valid_workers,
                shuffle=False)
dl_rev = DataLoader(ds_rev, 
                    batch_size=args.valid_batch_size, 
                    num_workers=args.valid_workers,
                    shuffle=False)
print("Evaluation...")
y = list()
y_true = list()
for its in dl:
    if type(its) in (tuple, list):
        x, yt = its
        x = x.to(device)
        y.extend(model(x)[-1].detach().cpu().flatten().tolist())
        y_true.extend(yt.detach().cpu().flatten().tolist())
    else:
        its = its.to(device)
        y.extend(model(its)[-1].detach().cpu().flatten().tolist())
y_rev = list()
for its in dl_rev:
    if type(its) in (tuple, list):
        x, yt = its
        x = x.to(device)
        y_rev.extend(model(x)[-1].detach().cpu().flatten().tolist())
    else:
        its = its.to(device)
        y_rev.extend(model(its)[-1].detach().cpu().flatten().tolist())
assert len(y) == len(y_rev)
for i in range(len(y)):
    y[i] = (y[i] + y_rev[i]) / 2
if args.output_format == 'json':
    keys = list(range(0, len(y)))
    d = dict(zip(keys, y))
    with open(args.output, 'w') as f:
        json.dump(d, f)
else:
    df = df_orig[['seq', 'bin']]
    df['bin'] = y
    df.to_csv(args.output, sep='\t' if args.output_format == 'tsv' else ',', index=None, header=False)
    
y = np.array(y)
if y_true:
    try:
        y_true = np.array(y_true)
        mse = np.mean((y_true - y) ** 2)
        r_pearson = pearsonr(y, y_true)
        r_spearman = spearmanr(y, y_true)
        print(f'MSE: {mse}, Pearson: {r_pearson}, Spearman: {r_spearman}')
    except ValueError:
        pass
