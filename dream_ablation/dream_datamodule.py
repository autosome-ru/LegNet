from curses import use_default_colors
import pandas as pd 
from pathlib import Path

import lightning.pytorch as pl

from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from dream_preprocess import (preprocess_data, 
                              add_singleton_column, 
                              add_rev, 
                              add_fold_column, 
                              make_rev)
from dream_dataset import SeqDataset
from dream_lightning_wrapper import LegNetConfig

class DreamTrainValDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_path: str,
                 valid_folds: list[int],
                 add_single_column: bool, 
                 reverse_augment: bool, 
                 seqsize: int = 150,
                 shift: float = 0.5,
                 scale: float = 0.5,
                 train_batch_size: int=1024,
                 valid_batch_size: int=4096,
                 num_workers: int = 16,
                 hash_seed: int =42):
        super().__init__()
        self.dataset_path = dataset_path
        self.valid_folds = valid_folds
        self.seqsize = seqsize
        self.shift = shift
        self.scale = scale
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        self.add_single_column = add_single_column
        self.reverse_augment = reverse_augment
        self.hash_seed = hash_seed

    def setup(self, stage: str):
        dataset = pd.read_table(self.dataset_path, 
                                    sep='\t', 
                                    header=None)
        dataset.columns = ['seq', 'bin', 'fold'][:len(dataset.columns)]
        dataset = preprocess_data(dataset, self.seqsize)
        if self.add_single_column:
            dataset = add_singleton_column(dataset)
        dataset = add_fold_column(dataset, self.hash_seed)
        
        valid_mask = dataset.fold.isin(self.valid_folds)
        train = dataset[~valid_mask]
        valid = dataset[valid_mask]
        if self.reverse_augment: 
            self.train = add_rev(train, mode="train")
            self.valid = add_rev(valid, mode="valid")
        
        self.train_ds = SeqDataset(self.train, 
                         size=self.seqsize, 
                         add_single_channel=self.add_single_column,
                         add_reverse_channel=self.reverse_augment,
                         return_probs=True,
                         shift=self.shift,
                         scale=self.scale)
                
        self.valid_ds = SeqDataset(self.valid, 
                         size=self.seqsize, 
                         add_single_channel=self.add_single_column,
                         add_reverse_channel=self.reverse_augment,
                         return_probs=False)
    
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, 
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

class DreamFullTrainDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_path: str,
                 add_single_column: bool, 
                 reverse_augment: bool, 
                 seqsize: int = 150,
                 shift: float = 0.5,
                 scale: float = 0.5,
                 train_batch_size: int=1024,
                 num_workers: int = 16):
        super().__init__()
        self.dataset_path = dataset_path
        self.seqsize = seqsize
        self.shift = shift
        self.scale = scale
        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.add_single_column = add_single_column
        self.reverse_augment = reverse_augment
        self.save_hyperparameters()   
    
    def setup(self, stage: str):
        dataset = pd.read_table(self.dataset_path, 
                                    sep='\t', 
                                    header=None)
        dataset.columns = ['seq', 'bin', 'fold'][:len(dataset.columns)]
        dataset = preprocess_data(dataset, self.seqsize)
        dataset = add_singleton_column(dataset)
        if self.reverse_augment: 
            self.dataset = add_rev(dataset, mode='train')
        self.full_ds = SeqDataset(self.dataset, 
                         size=self.seqsize, 
                         add_single_channel=self.add_single_column,
                         add_reverse_channel=self.reverse_augment,
                         return_probs=True,
                         shift=self.shift,
                         scale=self.scale,)
    
    def train_dataloader(self):
        return DataLoader(self.full_ds, 
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)


@dataclass
class DreamDataModuleConfig:
    split_type: str
    dataset_path: str
    add_single_column: bool
    reverse_augment: bool 
    seqsize: int 
    shift: float
    scale: float
    train_batch_size: int
    valid_batch_size: int
    num_workers: int 
    valid_folds: list[int] = field(default_factory=list)
    fold_hash_seed: int = 42
    
    @property
    def validation_exists(self) -> bool:
        if self.split_type == "fulltrain":
            return False
        else:
            return True
    
    def get_datamodule(self) -> pl.LightningDataModule:
        if self.split_type == "fulltrain":
            return DreamFullTrainDataModule(dataset_path=self.dataset_path,
                                            add_single_column=self.add_single_column,
                                            reverse_augment=self.reverse_augment,
                                            seqsize=self.seqsize,
                                            shift=self.shift,
                                            scale=self.scale,
                                            train_batch_size=self.train_batch_size,
                                            num_workers=self.num_workers)
        elif self.split_type == "trainvalid":
            assert len(self.valid_folds) > 0, "For trainvalid split type at least one validation fold required"
            return DreamTrainValDataModule(dataset_path=self.dataset_path,
                                           valid_folds=self.valid_folds,
                                           add_single_column=self.add_single_column,
                                           reverse_augment=self.reverse_augment,
                                           seqsize=self.seqsize,
                                           shift=self.shift,
                                           scale=self.scale,
                                           train_batch_size=self.train_batch_size,
                                           valid_batch_size=self.valid_batch_size,
                                           num_workers=self.num_workers,
                                           hash_seed=self.fold_hash_seed)
        else:
            raise Exception("Wrong datamodule type")
    
    def dls_for_predictions(self,
                        seqs: str | Path,
                        model_cfg: LegNetConfig,
                        singleton_mode: str):
        
        seq_df = pd.read_csv(seqs, 
                             sep="\t",
                             header=None)   
        assert len(seq_df.columns) < 3, "sequence table must contain at most 2 columns - seq and bin"
        
        seq_df.columns = ['seq', 'bin'][:len(seq_df.columns)]
        initial_df = seq_df.copy()
        seq_df = preprocess_data(seq_df, self.seqsize)
        seq_df = add_singleton_column(seq_df, mode=singleton_mode)
        
        return initial_df, self._gen_dls_for_predictions(seq_df=seq_df,
                                                     model_cfg=model_cfg)
        
    def _make_pred_dl(self, df: pd.DataFrame, model_cfg: LegNetConfig, rev: int):
        df = df.copy()
        df['rev'] = rev
        ds = SeqDataset(df,
                             size=self.seqsize, 
                             add_single_channel=self.add_single_column,
                             add_reverse_channel=model_cfg.use_reverse_channel,
                             return_probs=False)
        dl = DataLoader(dataset=ds, 
                        batch_size=self.valid_batch_size,
                        num_workers=self.num_workers,
                        shuffle=False)
        return dl
        
    def _gen_dls_for_predictions(self,
                                 seq_df: pd.DataFrame, 
                                 model_cfg: LegNetConfig):
        forw_dl = self._make_pred_dl(df=seq_df, model_cfg=model_cfg, rev=0)
        yield "forw", forw_dl
       
        if self.reverse_augment:
            rev_df = make_rev(seq_df)  
            rev_dl = self._make_pred_dl(df=rev_df,
                                        model_cfg=model_cfg, 
                                        rev=1)
            yield "rev", rev_dl
            
        
        
        
        
        
       
    