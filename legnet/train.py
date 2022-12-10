# -*- coding: utf-8 -*-
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import ClassVar, Iterator, Sequence

import mmh3
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError, Metric
from scipy.stats import pearsonr, spearmanr
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from model import SeqNN

LEFT_ADAPTER = "TGCATTTTTTTCACATC" 
RIGHT_ADAPTER = "GGTTACGGCTGTT"

PLASMID = "aactctcaaggatcttaccgctgttgagatccagttcgatgtaacccactcgtgcacccaactgatcttcagcatcttttactttcaccagcgtttctgggtgagcaaaaacaggaaggcaaaatgccgcaaaaaagggaataagggcgacacggaaatgttgaatactcatactcttcctttttcaatattattgaagcatttatcagggttattgtctcatgagcggatacatatttgaatgtatttagaaaaataaacaaataggggttccgcgcacatttccccgaaaagtgccacctgacgtcatctatattaccctgttatccctagcggatctgccggtagaggtgtggtcaataagagcgacctcatactatacctgagaaagcaacctgacctacaggaaagagttactcaagaataagaattttcgttttaaaacctaagagtcactttaaaatttgtatacacttattttttttataacttatttaataataaaaatcataaatcataagaaattcgcttatttagaagtGGCGCGCCGGTCCGttacttgtacagctcgtccatgccgccggtggagtggcggccctcggcgcgttcgtactgttccacgatggtgtagtcctcgttgtgggaggtgatgtccaacttgatgttgacgttgtaggcgccgggcagctgcacgggcttcttggccttgtaggtggtcttgacctcagcgtcgtagtggccgccgtccttcagcttcagcctctgcttgatctcgcccttcagggcgccgtcctcggggtacatccgctcggaggaggcctcccagcccatggtcttcttctgcattacggggccgtcggaggggaagttggtgccgcgcagcttcaccttgtagatgaactcgccgtcctgcagggaggagtcctgggtcacggtcaccacgccgccgtcctcgaagttcatcacgcgctcccacttgaagccctcggggaaggacagcttcaagtagtcggggatgtcggcggggtgcttcacgtaggccttggagccgtacatgaactgaggggacaggatgtcccaggcgaagggcagggggccacccttggtcaccttcagcttggcggtctgggtgccctcgtaggggcggccctcgccctcgccctcgatctcgaactcgtggccgttcacggagccctccatgtgcaccttgaagcgcatgaactccttgatgatggccatgttatcctcctcgcccttgctcacCATGGTACTAGTGTTTAGTTAATTATAGTTCGTTGACCGTATATTCTAAAAACAAGTACTCCTTAAAAAAAAACCTTGAAGGGAATAAACAAGTAGAATAGATAGAGAGAAAAATAGAAAATGCAAGAGAATTTATATATTAGAAAGAGAGAAAGAAAAATGGAAAAAAAAAAATAGGAAAAGCCAGAAATAGCACTAGAAGGAGCGACACCAGAAAAGAAGGTGATGGAACCAATTTAGCTATATATAGTTAACTACCGGCTCGATCATCTCTGCCTCCAGCATAGTCGAAGAAGAATTTTTTTTTTCTTGAGGCTTCTGTCAGCAACTCGTATTTTTTCTTTCTTTTTTGGTGAGCCTAAAAAGTTCCCACGTTCTCTTGTACGACGCCGTCACAAACAACCTTATGGGTAATTTGTCGCGGTCTGGGTGTATAAATGTGTGGGTGCAACATGAATGTACGGAGGTAGTTTGCTGATTGGCGGTCTATAGATACCTTGGTTATGGCGCCCTCACAGCCGGCAGGGGAAGCGCCTACGCTTGACATCTACTATATGTAAGTATACGGCCCCATATATAggccctttcgtctcgcgcgtttcggtgatgacggtgaaaacctctgacacatgcagctcccggagacggtcacagcttgtctgtaagcggatgccgggagcagacaagcccgtcagggcgcgtcagcgggtgttggcgggtgtcggggctggcttaactatgcggcatcagagcagattgtactgagagtgcaccatatggacatattgtcgttagaacgcggctacaattaatacataaccttatgtatcatacacatacgatttaggtgacactatagaacgcggccgccagctgaagctttaactatgcggcatcagagcagattgtactgagagtgcaccataccaccttttcaattcatcattttttttttattcttttttttgatttcggtttccttgaaatttttttgattcggtaatctccgaacagaaggaagaacgaaggaaggagcacagacttagattggtatatatacgcatatgtagtgttgaagaaacatgaaattgcccagtattcttaacccaactgcacagaacaaaaacctgcaggaaacgaagataaatcatgtcgaaagctacatataaggaacgtgctgctactcatcctagtcctgttgctgccaagctatttaatatcatgcacgaaaagcaaacaaacttgtgtgcttcattggatgttcgtaccaccaaggaattactggagttagttgaagcattaggtcccaaaatttgtttactaaaaacacatgtggatatcttgactgatttttccatggagggcacagttaagccgctaaaggcattatccgccaagtacaattttttactcttcgaagacagaaaatttgctgacattggtaatacagtcaaattgcagtactctgcgggtgtatacagaatagcagaatgggcagacattacgaatgcacacggtgtggtgggcccaggtattgttagcggtttgaagcaggcggcagaagaagtaacaaaggaacctagaggccttttgatgttagcagaattgtcatgcaagggctccctatctactggagaatatactaagggtactgttgacattgcgaagagcgacaaagattttgttatcggctttattgctcaaagagacatgggtggaagagatgaaggttacgattggttgattatgacacccggtgtgggtttagatgacaagggagacgcattgggtcaacagtatagaaccgtggatgatgtggtctctacaggatctgacattattattgttggaagaggactatttgcaaagggaagggatgctaaggtagagggtgaacgttacagaaaagcaggctgggaagcatatttgagaagatgcggccagcaaaactaaaaaactgtattataagtaaatgcatgtatactaaactcacaaattagagcttcaatttaattatatcagttattaccctatgcggtgtgaaataccgcacagatgcgtaaggagaaaataccgcatcaggaaattgtaagcgttaatattttgttaaaattcgcgttaaatttttgttaaatcagctcattttttaaccaataggccgaaatcggcaaaatcccttataaatcaaaagaatagaccgagatagggttgagtgttgttccagtttggaacaagagtccactattaaagaacgtggactccaacgtcaaagggcgaaaaaccgtctatcagggcgatggcccactacgtgaaccatcaccctaatcaagtGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGGTTACGGCTGTTTCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGAAGGCAAAGatgtctaaaggtgaagaattattcactggtgttgtcccaattttggttgaattagatggtgatgttaatggtcacaaattttctgtctccggtgaaggtgaaggtgatgctacttacggtaaattgaccttaaaattgatttgtactactggtaaattgccagttccatggccaaccttagtcactactttaggttatggtttgcaatgttttgctagatacccagatcatatgaaacaacatgactttttcaagtctgccatgccagaaggttatgttcaagaaagaactatttttttcaaagatgacggtaactacaagaccagagctgaagtcaagtttgaaggtgataccttagttaatagaatcgaattaaaaggtattgattttaaagaagatggtaacattttaggtcacaaattggaatacaactataactctcacaatgtttacatcactgctgacaaacaaaagaatggtatcaaagctaacttcaaaattagacacaacattgaagatggtggtgttcaattagctgaccattatcaacaaaatactccaattggtgatggtccagtcttgttaccagacaaccattacttatcctatcaatctgccttatccaaagatccaaacgaaaagagagaccacatggtcttgttagaatttgttactgctgctggtattacccatggtatggatgaattgtacaaataaggcgcgccacttctaaataagcgaatttcttatgatttatgatttttattattaaataagttataaaaaaaataagtgtatacaaattttaaagtgactcttaggttttaaaacgaaaattcttattcttgagtaactctttcctgtaggtcaggttgctttctcaggtatagtatgaggtcgctcttattgaccacacctctaccggcagatccgctagggataacagggtaatataGATCTGTTTAGCTTGCCTCGTCCCCGCCGGGTCACCCGGCCAGCGACATGGAGGCCCAGAATACCCTCCTTGACAGTCTTGACGTGCGCAGCTCAGGGGCATGATGTGACTGTCGCCCGTACATTTAGCCCATACATCCCCATGTATAATCATTTGCATCCATACATTTTGATGGCCGCACGGCGCGAAGCAAAAATTACGGCTCCTCGCTGCAGACCTGCGAGCAGGGAAACGCTCCCCTCACAGACGCGTTGAATTGTCCCCACGCCGCGCCCCTGTAGAGAAATATAAAAGGTTAGGATTTGCCACTGAGGTTCTTCTTTCATATACTTCCTTTTAAAATCTTGCTAGGATACAGTTCTCACATCACATCCGAACATAAACAACCATGGGTACCACTCTTGACGACACGGCTTACCGGTACCGCACCAGTGTCCCGGGGGACGCCGAGGCCATCGAGGCACTGGATGGGTCCTTCACCACCGACACCGTCTTCCGCGTCACCGCCACCGGGGACGGCTTCACCCTGCGGGAGGTGCCGGTGGACCCGCCCCTGACCAAGGTGTTCCCCGACGACGAATCGGACGACGAATCGGACGACGGGGAGGACGGCGACCCGGACTCCCGGACGTTCGTCGCGTACGGGGACGACGGCGACCTGGCGGGCTTCGTGGTCGTCTCGTACTCCGGCTGGAACCGCCGGCTGACCGTCGAGGACATCGAGGTCGCCCCGGAGCACCGGGGGCACGGGGTCGGGCGCGCGTTGATGGGGCTCGCGACGGAGTTCGCCCGCGAGCGGGGCGCCGGGCACCTCTGGCTGGAGGTCACCAACGTCAACGCACCGGCGATCCACGCGTACCGGCGGATGGGGTTCACCCTCTGCGGCCTGGACACCGCCCTGTACGACGGCACCGCCTCGGACGGCGAGCAGGCGCTCTACATGAGCATGCCCTGCCCCTAATCAGTACTGACAATAAAAAGATTCTTGTTTTCAAGAACTTGTCATTTGTATAGTTTTTTTATATTGTAGTTGTTCTATTTTAATCAAATGTTAGCGTGATTTATATTTTTTTTCGCCTCGACATCATCTGCCCAGATGCGAAGTTAAGTGCGCAGAAAGTAATATCATGCGTCAATCGTATGTGAATGCTGGTCGCTATACTGCTGTCGATTCGATACTAACGCCGCCATCCAGTGTCGAAAACGAGCTCGaattcctgggtccttttcatcacgtgctataaaaataattataatttaaattttttaatataaatatataaattaaaaatagaaagtaaaaaaagaaattaaagaaaaaatagtttttgttttccgaagatgtaaaagactctagggggatcgccaacaaatactaccttttatcttgctcttcctgctctcaggtattaatgccgaattgtttcatcttgtctgtgtagaagaccacacacgaaaatcctgtgattttacattttacttatcgttaatcgaatgtatatctatttaatctgcttttcttgtctaataaatatatatgtaaagtacgctttttgttgaaattttttaaacctttgtttatttttttttcttcattccgtaactcttctaccttctttatttactttctaaaatccaaatacaaaacataaaaataaataaacacagagtaaattcccaaattattccatcattaaaagatacgaggcgcgtgtaagttacaggcaagcgatccgtccGATATCatcagatccactagtggcctatgcggccgcggatctgccggtctccctatagtgagtcgtattaatttcgataagccaggttaacctgcattaatgaatcggccaacgcgcggggagaggcggtttgcgtattgggcgctcttccgcttcctcgctcactgactcgctgcgctcggtcgttcggctgcggcgagcggtatcagctcactcaaaggcggtaatacggttatccacagaatcaggggataacgcaggaaagaacatgtgagcaaaaggccagcaaaaggccaggaaccgtaaaaaggccgcgttgctggcgtttttccataggctccgcccccctgacgagcatcacaaaaatcgacgctcaagtcagaggtggcgaaacccgacaggactataaagataccaggcgtttccccctggaagctccctcgtgcgctctcctgttccgaccctgccgcttaccggatacctgtccgcctttctcccttcgggaagcgtggcgctttctcaTAgctcacgctgtaggtatctcagttcggtgtaggtcgttcgctccaagctgggctgtgtgcacgaaccccccgttcagcccgaccgctgcgccttatccggtaactatcgtcttgagtccaacccggtaagacacgacttatcgccactggcagcagccactggtaacaggattagcagagcgaggtatgtaggcggtgctacagagttcttgaagtggtggcctaactacggctacactagaagAacagtatttggtatctgcgctctgctgaagccagttaccttcggaaaaagagttggtagctcttgatccggcaaacaaaccaccgctggtagcggtggtttttttgtttgcaagcagcagattacgcgcagaaaaaaaggatctcaagaagatcctttgatcttttctacggggtctgacgctcagtggaacgaaaactcacgttaagggattttggtcatgagattatcaaaaaggatcttcacctagatccttttaaattaaaaatgaagttttaaatcaatctaaagtatatatgagtaaacttggtctgacagttaccaatgcttaatcagtgaggcacctatctcagcgatctgtctatttcgttcatccatagttgcctgactccccgtcgtgtagataactacgatacgggagggcttaccatctggccccagtgctgcaatgataccgcgagacccacgTtcaccggctccagatttatcagcaataaaccagccagccggaagggccgagcgcagaagtggtcctgcaactttatccgcctccatccagtctattaattgttgccgggaagctagagtaagtagttcgccagttaatagtttgcgcaacgttgttgccattgctacaggcatcgtggtgtcacgctcgtcgtttggtatggcttcattcagctccggttcccaacgatcaaggcgagttacatgatcccccatgttgtgcaaaaaagcggttagctccttcggtcctccgatcgttgtcagaagtaagttggccgcagtgttatcactcatggttatggcagcactgcataattctcttactgtcatgccatccgtaagatgcttttctgtgactggtgagtactcaaccaagtcattctgagaatagtgtatgcggcgaccgagttgctcttgcccggcgtcaatacgggataataccgcgccacatagcagaactttaaaagtgctcatcattggaaaacgttcttcggggcgaa"
PLASMID = PLASMID.upper()
INSERT_START = PLASMID.find('N'*80)

def hash_fun(seq, seed):
    return mmh3.hash(seq, seed, signed=False) % 10

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

class PearsonMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._ys = []
        self._ypreds = []
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._ys = []
        self._ypreds = []
        super().reset()

    def update(self, output):
        y_pred, y = output[0].cpu().numpy(), output[1].cpu().numpy()
        self._ys.append(y)
        self._ypreds.append(y_pred)
        
    def compute(self):
        y = np.concatenate(self._ys)
        y_pred = np.concatenate(self._ypreds)
        cor, _ = pearsonr(y, y_pred)
        return cor 

class SpearmanMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._ys = []
        self._ypreds = []
        super().__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._ys = []
        self._ypreds = []
        super().reset()

    def update(self, output):
        y_pred, y = output[0].cpu().numpy(), output[1].cpu().numpy()
        self._ys.append(y)
        self._ypreds.append(y_pred)
        
    def compute(self):
        y = np.concatenate(self._ys)
        y_pred = np.concatenate(self._ypreds)
        cor, _ = spearmanr(y, y_pred)
        return cor 


def create_trainer(model, 
                   optimizer,
                   scheduler,  
                   criterion, 
                   device, 
                   model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    train_mse =  MeanSquaredError()
    train_pearson = PearsonMetric()
    train_spearman = SpearmanMetric()
    
    def train_step(trainer, batch):
        nonlocal model
        if not model.training:
            model = model.train()
        X, y_probs, y = batch
        X = X.to(device)
        y_probs = y_probs.float().to(device)
        logprobs, y_pred = model(X)
        loss = criterion(logprobs, y_probs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()                                                                                         
        out = (y_pred.detach().cpu(), y)
        train_mse.update(out)
        train_pearson.update(out)
        train_spearman.update(out)
               
        return loss.item()
    
    trainer = Engine(train_step)
    
    @trainer.on(Events.STARTED)
    def prepare_epoch(engine):
        engine.state.metrics['train_pearson'] = -np.inf
        engine.state.metrics['train_mse'] = -np.inf
        engine.state.metrics['train_spearman'] = -np.inf

    def evaluate(engine, batch):
        nonlocal model
        if model.training:
            model = model.eval()
        with torch.no_grad():
            X, y = batch
            X = X.to(device)
            y = y.float().to(device)
            _, y_pred = model(X, predict_score=True)
        return y_pred.cpu(), y.cpu()

    evaluator = Engine(evaluate)

    MeanSquaredError().attach(evaluator, 'mse')
    p = PearsonMetric()
    p.attach(evaluator, 'pearson')
    s = SpearmanMetric()
    s.attach(evaluator, 'spearman')
    
    @trainer.on(Events.EPOCH_COMPLETED) #| Events.STARTED)
    def validate(engine):
        p.reset()
            
        engine.state.metrics['train_mse'] = train_mse.compute()
        engine.state.metrics['train_pearson'] = train_pearson.compute()
        engine.state.metrics['train_spearman'] = train_spearman.compute()
        train_mse.reset()
        train_pearson.reset()
        train_spearman.reset()
        
        score_path =  model_dir / f"scores_{engine.state.epoch}.json"
        with open(score_path, "w") as outp:
            json.dump(engine.state.metrics, outp)

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def dump_model(engine):
        model_path = model_dir / f"model_{engine.state.epoch}.pth"
        torch.save(model.state_dict(), model_path)
        
        optimizer_path = model_dir / f"optimizer_{engine.state.epoch}.pth"
        torch.save(optimizer.state_dict(), optimizer_path)
        
        if scheduler is not None:
            scheduler_path = model_dir / f"scheduler_{engine.state.epoch}.pth"
            torch.save(scheduler.state_dict(), scheduler_path)
           
            
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, ["train_mse", "train_pearson", "train_spearman", ], 
                output_transform=lambda x: {'batch_loss': x}, 
                )
    return trainer, p

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

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

parser.add_argument("--train_valid_path", default="train_val.tsv", help="path to a training dataset")
parser.add_argument("--seed", type=int, default=42, help='seed for pseudo-random number generators')
parser.add_argument("--train_batch_size", type=int, default=1024, help="batch size used at training stage")
parser.add_argument("--train_workers", type=int, default=8, help="number of workers used to read/postprocess data during training")
parser.add_argument("--valid_batch_size", type=int, default=4098, help="batch size used at validation stage")
parser.add_argument("--valid_workers", type=int, default=8, help="number of workers used to read/postprocess data during validation")
parser.add_argument("--use_sampler", action="store_true", help="use weighted sampler")
parser.add_argument("--epoch_num", type=int, default=80, help="number of training epochs")
parser.add_argument("--batch_per_epoch", type=int, default=1000, help="epoch is defines as batch_per_epoch batches")
parser.add_argument("--weights", choices=["uniform", "counts"], default="uniform")
parser.add_argument("--seqsize", type=int, default=120, help="sequences will be padded so their length is equal to seqsize")
parser.add_argument("--temp", default=".TEMPDIR", type=Path, help="directory name for auxilarily files")
parser.add_argument("--use_single_channel", action="store_true", help="use an extra channel to encode singleton information")
parser.add_argument("--singleton_definition", choices=["integer", "threshold1100"], default="integer", help="singleton mode")
parser.add_argument("--gpu", type=int, default=0, help="ID/number of a GPU device that is used")
parser.add_argument("--model_dir", type=Path, required=True, help="directory name where training results are saved")
parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
parser.add_argument("--ks", default=5, type=int, help="kernel size of convolutional layers")
parser.add_argument("--blocks", default=[256, 256, 128, 128, 64, 64, 32, 32], nargs="+", type=int, help="number of channels for EffNet-like blocks")
parser.add_argument("--resize_factor", default=4, type=int, help="number of channels in a middle/high-dimensional convolutional layer of an EffNet-like block")
parser.add_argument("--se_reduction", default=4, type=float, help="reduction number used in SELayer")
parser.add_argument("--shift", default=0.5, type=float)
parser.add_argument("--scale", default=1.5, type=str)
parser.add_argument("--loss", choices=["mse", "kl"], default="mse", type=str, help="loss function")
parser.add_argument("--final_ch", default=18, type=int, help="number of channels of the final convolutional layer")
parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw", "rmsprop"], help="optimizer name")
parser.add_argument("--scheduler", default="onecycle", choices=["onecycle"], help="scheduler used during optimization")
parser.add_argument("--div_factor", default=25.0, type=float)
parser.add_argument("--max_lr", default=0.005, type=float)
parser.add_argument("--pct_start", default=0.3, type=float)
parser.add_argument("--three_phase", action="store_true")
parser.add_argument("--float32_approx", action="store_true")
parser.add_argument("--bn_momentum", default=0.1, type=float)
parser.add_argument("--delimiter", default='space', type=str, help="delimiter that separates columns in a training file")
parser.add_argument("--foldify", action="store_true")

args = parser.parse_args()
print(args)
torch.backends.cuda.matmul.allow_tf32 = args.float32_approx # default is False

if args.use_sampler and args.weights is None:
    args.weights=None

args.model_dir.mkdir(exist_ok=False, parents=True)
args_path = args.model_dir / "args.json"

with args_path.open('w') as outp:
    dt = {x: str(y) if isinstance(y, Path) else y for x, y in vars(args).items()}
    json.dump(dt, outp, indent=4)
    
run_backup_path = args.model_dir / "run.py"

shutil.copy(sys.argv[0], run_backup_path)
shutil.copy(os.path.join(os.path.dirname(sys.argv[0]), 'model.py'), os.path.join(args.model_dir, 'model.py'))


if not args.temp.exists():
    args.temp.mkdir()

train_path  = args.temp / f"train_{args.seqsize}_{args.singleton_definition}_from_{Path(args.train_valid_path).stem}.txt"
valid_path = args.temp / f"valid_{args.seqsize}_{args.singleton_definition}_from_{Path(args.train_valid_path).stem}.txt"

if not (train_path.exists() and valid_path.exists()):
    train_valid = pd.read_table(args.train_valid_path, 
         sep='\t' if args.delimiter == 'tab' else ' ', 
         header=None) # modified
    err_str = f"No bin column in a training dataset! Make sure that the --delimiter argument is correct (tab or space, current: {args.delimiter})."
    assert len(train_valid.columns) >= 2, err_str
    train_valid.columns = ['seq', 'bin', 'fold'][:len(train_valid.columns)]
    if args.foldify and ('fold' not in train_valid):
        fold = list(map(lambda x: hash_fun(x, args.seed), train_valid.seq))
        train_valid['fold'] = fold
        train_valid = train_valid.sort_values('fold')

    print(train_valid.head())

    train_valid = preprocess_data(train_valid, args.seqsize)
    if args.use_single_channel:
        train_valid = add_singleton_column(train_valid, args.singleton_definition)

    train = train_valid

    train = add_rev(train)
    train.to_csv(train_path, sep="\t", index=False, header=True)
else:
    train = pd.read_table(train_path)

train_ds = SeqDatasetRevProb(train, 
                         size=args.seqsize, 
                         use_single_channel=args.use_single_channel,
                         shift=args.shift,
                         scale=args.scale)



set_global_seed(args.seed)

if args.use_sampler:
    print("Using custom sampler")
    weights = get_weights(train, args.weights)
    train_per_epoch_size = args.train_batch_size * args.batch_per_epoch
    sampler=CustomWeightedRandomSampler(weights=weights,
                                        num_samples=train_per_epoch_size)
    train_dl = DataLoader(train_ds, 
                          batch_size=args.train_batch_size,
                          num_workers=args.train_workers,
                          sampler=sampler, 
                          shuffle=False)                                       
else:
    print("Using shuffle")
    train_dl = DataLoader(train_ds, 
                          batch_size=args.train_batch_size,
                          num_workers=args.train_workers,
                          shuffle=True)    
    train_dl = DataloaderWrapper(train_dl, args.batch_per_epoch)
    
device = torch.device(f"cuda:{args.gpu}")
model = SeqNN(seqsize=args.seqsize, 
              use_single_channel=args.use_single_channel,
              block_sizes= args.blocks, 
              ks=args.ks, 
              resize_factor=args.resize_factor, 
              se_reduction=args.se_reduction, 
              bn_momentum=args.bn_momentum,
              final_ch=args.final_ch).to(device)

model.apply(initialize_weights)

min_lr = args.max_lr / args.div_factor

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr = min_lr, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr = min_lr, weight_decay=args.weight_decay)
elif args.optimizer == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr = min_lr, weight_decay=args.weight_decay)
else:
    raise Exception("Wrong optimizer")
    
if args.loss == "kl":
    criterion = nn.KLDivLoss( reduction= "batchmean").to(device)
elif args.loss == "mse":
    criterion = nn.MSELoss().to(device)
else:
    raise Exception("Wrong loss")
    
model_dir = args.model_dir
log_dir = model_dir / "logs"

if args.scheduler == "onecycle":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=args.max_lr,
                                                    steps_per_epoch=args.batch_per_epoch, 
                                                    epochs=args.epoch_num, 
                                                    pct_start=args.pct_start,
                                                    three_phase=args.three_phase, 
                                                    div_factor=args.div_factor)
else:
    raise Exception("Wrong scheduler type")


print('Parameters:', int(parameter_count(model)))
trainer, p = create_trainer(model, optimizer, scheduler, criterion, device, 
                           model_dir)

tb_logger = TensorboardLogger(log_dir=log_dir)
tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
)

state = trainer.run(train_dl, max_epochs=args.epoch_num) # type: ignore
