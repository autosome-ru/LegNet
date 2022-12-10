# -*- coding: utf-8 -*-
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import ClassVar, Iterator, Sequence
import json

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
from ignite.handlers import FastaiLRFinder
from ignite.metrics import MeanSquaredError, Metric
from scipy.stats import pearsonr, spearmanr
from tltorch import TRL
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler


def hash_fun(seq, seed):
    return mmh3.hash(seq, seed, signed=False) % 10

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

def parameter_count(model):
    pars = 0  
    for _, p  in model.named_parameters():    
        pars += torch.prod(torch.tensor(p.shape))
    return pars

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

def add_rev(df):
    df = df.copy()
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    df['rev'] = 0
    revdf['rev'] = 1
    df = pd.concat([df, revdf]).reset_index(drop=True)
    return df

def infer_singleton(arr, method="integer"):
    if method == "integer":
        return np.array([x.is_integer() for x in arr])
    elif method.startswith("threshold"):
        th = float(method.replace("threshold", ""))
        cnt = Counter(arr)
        return np.array([cnt[x] >= th for x in arr])
    else:
        raise Exception("Wrong method")

def add_singleton_column(df, method="integer"):
    df = df.copy()
    df["is_singleton"] = infer_singleton(df.bin.values,method)
    return df 

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

def cash_and_preprocess(seqsize, path_to_training_data, delimiter, foldify, use_single_channel, use_reverse_channel, use_multisubstate_channel, preprocess_data, use_validation, seed):
    temp = Path('.TEMPDIR')
    if not temp.exists():
        temp.mkdir()
    temp_val = 'part' if use_validation else 'full'

    # save preprocessed dataset to not repreprocess it
    train_path  = temp / f"train_{seqsize}_from_{Path(path_to_training_data).stem}_{temp_val}.txt"
    valid_path = temp / f"valid_{seqsize}_from_{Path(path_to_training_data).stem}_{temp_val}.txt"

    if not (train_path.exists() and valid_path.exists()):
        
        train_valid = pd.read_table(path_to_training_data, sep='\t' if delimiter == 'tab' else ' ', header=None) 
        
        err_str = f"No bin column in a training dataset!\n"
        err_str += f"Make sure that the --delimiter argument is correct (tab or space, current: {delimiter})."
        assert len(train_valid.columns) >= 2, err_str
        
        if not use_multisubstate_channel:
            train_valid.columns = ['seq', 'bin', 'fold'][:len(train_valid.columns)]
        else:
            train_valid.columns = ['seq', 'bin', 'substrate', 'fold'][:len(train_valid.columns)]
        
        if foldify and ('fold' not in train_valid):
            fold = list(map(lambda x: hash_fun(x, seed), train_valid.seq))
            train_valid['fold'] = fold
            train_valid = train_valid.sort_values('fold')

        print(train_valid.head())

        train_valid = preprocess_data(train_valid, seqsize)
        if use_single_channel:
            train_valid = add_singleton_column(train_valid)
            
        if use_reverse_channel:
            train_valid = add_rev(train_valid)
            
        if not use_validation:
            train = train_valid
            valid = None
        else:
            train = train_valid[train_valid['fold'] != 9]
            valid = train_valid[train_valid['fold'] == 9]

        train.to_csv(train_path, sep="\t", index=False, header=True)
        if use_validation:
            valid.to_csv(valid_path, sep="\t", index=False, header=True)
    else:
        train = pd.read_table(train_path)
        if use_validation:
            valid = pd.read_table(valid_path)

    return train, valid

def create_dl(train, valid, 
              seqsize,
              use_single_channel, use_reverse_channel, use_multisubstate_channel,
              train_batch_size, train_workers,
              valid_batch_size, valid_workers,
              batch_per_epoch,
              SeqDatasetProb,
              shuffle_train=True, shuffle_val=False):
    
    train_ds = SeqDatasetProb(train, 
                             seqsize=seqsize, 
                             use_single_channel=use_single_channel,
                             use_reverse_channel=use_reverse_channel,
                             use_multisubstate_channel=use_multisubstate_channel,
                             )

    train_dl = DataLoader(train_ds, 
                          batch_size=train_batch_size,
                          num_workers=train_workers,
                          shuffle=shuffle_train) 

    train_dl = DataloaderWrapper(train_dl, batch_per_epoch)
    
    if valid is None:
        return train_dl, None
    
    valid_ds = SeqDatasetProb(valid, 
                            seqsize=seqsize, 
                            use_single_channel=use_single_channel,
                            use_reverse_channel=use_reverse_channel,
                            use_multisubstate_channel=use_multisubstate_channel
                            )

    valid_dl = DataLoader(valid_ds, 
                        batch_size=valid_batch_size,
                        num_workers=valid_workers,
                        shuffle=shuffle_val) 
    
    return train_dl, valid_dl

def get_model(SeqNN, seqsize, use_single_channel, use_reverse_channel,use_multisubstate_channel,
              blocks, ks, resize_factor, se_reduction, bn_momentum, final_ch, device):

    model = SeqNN(seqsize=seqsize, 
                  use_single_channel=use_single_channel,
                  use_reverse_channel=use_reverse_channel,
                  use_multisubstate_channel=use_multisubstate_channel,
                  block_sizes= blocks, 
                  ks=ks, 
                  resize_factor=resize_factor, 
                  se_reduction=se_reduction, 
                  bn_momentum=bn_momentum,
                  final_ch=final_ch).to(device)

    model.apply(initialize_weights)
    return model

def run_lr_finder(model, optimizer, criterion, train_dl, device):
    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}
    def train_step(model, trainer, batch):
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
        return loss.item()
        
        
    trainer = Engine(lambda trainer, batch: train_step(model, trainer, batch))

    with lr_finder.attach(trainer, to_save=to_save, num_iter=5000, start_lr=1e-6, end_lr=100, step_mode="exp") as trainer_with_lr_finder:
        trainer_with_lr_finder.run(train_dl)

    lr_finder.get_results()

    lr_finder.plot()
   
    suggestion = lr_finder.lr_suggestion()

    print(f'lr_finder suggestion: {suggestion}')

def get_optimizer(optimizer_name, model_params, lr, weight_decay):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model_params, lr = lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model_params, lr = lr, weight_decay=weight_decay)
    else:
        raise Exception("Wrong optimizer")
    return optimizer

def test_results(target, output, model, model_dir, epoch_num, seqsize, use_single_channel, use_reverse_channel, use_multisubstate_channel, preprocess_data, SeqDatasetProb, valid_batch_size, valid_workers, device):
    model.load_state_dict(torch.load(Path(model_dir) / f'model_{epoch_num}.pth'))
    model.eval()

    df = pd.read_table(target, sep='\t', header=None)
    if not use_multisubstate_channel:
        df.columns = ['seq', 'bin', 'fold'][:len(df.columns)]
    else:
        df.columns = ['seq', 'bin', 'substrate', 'fold'][:len(df.columns)]

    df = preprocess_data(df, seqsize)

    if use_single_channel:
        df = add_singleton_column(df)
    if use_reverse_channel:
        df_rev = get_rev(df)
        df['rev'] = 0

    ds = SeqDatasetProb(ds=df, seqsize=seqsize, 
                        use_single_channel=use_single_channel,
                        use_reverse_channel=use_reverse_channel,
                        use_multisubstate_channel=use_multisubstate_channel
                    )
    dl = DataLoader(ds, 
                    batch_size=valid_batch_size, 
                    num_workers=valid_workers,
                    shuffle=False)
    if use_reverse_channel:
        ds_rev = SeqDatasetProb(ds=df_rev, seqsize=seqsize, 
                                use_single_channel=use_single_channel,
                                use_reverse_channel=use_reverse_channel,
                                use_multisubstate_channel=use_multisubstate_channel)
        dl_rev = DataLoader(ds_rev, 
                            batch_size=valid_batch_size, 
                            num_workers=valid_workers,
                            shuffle=False)

    y = list()
    y_true = list()
    for its in dl:
        if type(its) in (tuple, list):
            x, probs, yt = its
            x = x.to(device)
            y.extend(model(x)[-1].detach().cpu().flatten().tolist())
            y_true.extend(yt.detach().cpu().flatten().tolist())
        else:
            its = its.to(device)
            y.extend(model(its)[-1].detach().cpu().flatten().tolist())

    if use_reverse_channel:
        y_rev = list()
        for its in dl_rev:
            if type(its) in (tuple, list):
                x, probs, yt = its
                x = x.to(device)
                y_rev.extend(model(x)[-1].detach().cpu().flatten().tolist())
            else:
                its = its.to(device)
                y_rev.extend(model(its)[-1].detach().cpu().flatten().tolist())
        assert len(y) == len(y_rev)
        for i in range(len(y)):
            y[i] = (y[i] + y_rev[i]) / 2

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

    df = pd.DataFrame({'seq': df.seq, 'bin': y})
    df.to_csv(output, sep='\t', index=None, header=False)