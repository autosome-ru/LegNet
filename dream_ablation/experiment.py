import glob
import json
from typing import ClassVar
import torch

import pandas as pd

from pathlib import Path
from dataclasses import dataclass, asdict 
from train_utils import set_global_seed
from dream_lightning_wrapper import DreamModel, LegNetConfig, TrainingSchemeConfig
from dream_datamodule import DreamDataModuleConfig
from dream_run import RunConfig


@dataclass
class Experiment:
    model_cfg: LegNetConfig
    train_scheme_cfg: TrainingSchemeConfig
    data_cfg: DreamDataModuleConfig
    run_cfg: RunConfig
    global_seed: int 
    set_medium_float32_precision: bool = True 
    train_log_step: int = 1000
    
    CONFIG_PATH_NAME: ClassVar[str] = 'params.json'
    MODELS_PATH: ClassVar[str] = 'lightning_logs/version_0/checkpoints/'
    LAST_MODEL_PREF: ClassVar[str] = 'last_model' # this can be refactored but not now
    BEST_MODEL_PREF: ClassVar[str] = 'model_'
    
    def run(self):
        self.check_params()
        self.root_dir.mkdir(parents=True)
        self.dump()
        set_global_seed(self.global_seed)
        if self.set_medium_float32_precision:
            torch.set_float32_matmul_precision('medium') 
        model = self.get_lt_model()
        
        data = self.data_cfg.get_datamodule(self.model_cfg)
        trainer = self.run_cfg.get_trainer(validation_exists=self.data_cfg.validation_exists,
                                           save_every_n_train_steps=self.train_log_step)
        trainer.fit(model, data)
        
    def predict(self, 
                seqs: str| Path,
                model_path: str | Path,
                singleton_mode: str,
                logdir: str,
                total_pred_col: str="bin"):
        if self.set_medium_float32_precision:
            torch.set_float32_matmul_precision('medium') 
        model = self.get_lt_model_from_checkpoint(checkpoint_path=model_path)
       
        
        seq_df, dl_iter = self.data_cfg.dls_for_predictions(seqs,
                                                   self.model_cfg,
                                                   singleton_mode=singleton_mode)
        runner = self.run_cfg.get_predict_runner(logdir)
        pred_cols = []
        for name, dl in dl_iter:
            predictions = runner.predict(model, dl)
            predictions = torch.concat(predictions) # type: ignore
            predictions = predictions.numpy()
            seq_df[name] = predictions
            pred_cols.append(name)
        
        seq_df[total_pred_col] = seq_df[pred_cols].mean(axis=1)    
        return seq_df

        
    def get_lt_model(self):
        model = DreamModel(model_cfg=self.model_cfg,
                   train_cfg=self.train_scheme_cfg,
                   train_log_step=self.train_log_step,
                   for_lr_finder=False)
        return model
    
    def get_lt_model_from_checkpoint(self, checkpoint_path: str | Path):
        model = DreamModel.load_from_checkpoint(
                   checkpoint_path=checkpoint_path,
                   model_cfg=self.model_cfg,
                   train_cfg=self.train_scheme_cfg,
                   train_log_step=self.train_log_step,
                   for_lr_finder=False)
        return model

    def last_model_path(self, exp_root_path: str | Path) -> Path:
        #Can't use self.root_path as paths may have changed
        
        exp_root_path = Path(exp_root_path)
        pat = str(exp_root_path / self.MODELS_PATH / f"{self.LAST_MODEL_PREF}*")
        paths = glob.glob(pat)
        if len(paths) == 0:
            raise Exception(f"No last models found: {pat}")
        if len(paths) > 1:
            raise Exception(f"More than 1 last model found: {pat}")
        path = Path(paths[0])
        return path
    
    def best_model_path(self, exp_root_path: str | Path) -> Path:
        #Can't use self.root_path as paths may have changed
        exp_root_path = Path(exp_root_path)
        pat = str(exp_root_path / self.MODELS_PATH / f"{self.BEST_MODEL_PREF}*")
        paths = glob.glob(pat)
        if len(paths) == 0:
            raise Exception(f"No best models found: {pat}")
        if len(paths) > 1:
            raise Exception(f"More than 1 best models found: {pat}")
        path = Path(paths[0])
        return path

    @property
    def root_dir(self) -> Path:
        return Path(self.run_cfg.model_dir)
    
    @property
    def experiment_config_path(self) -> Path:
        return self.root_dir / self.CONFIG_PATH_NAME
    
    def check_params(self): 
        if self.root_dir.exists():
            raise Exception(f"Model dir already exists: {self.root_dir}")
        if not self.data_cfg.reverse_augment:
            if self.model_cfg.use_reverse_channel:
                raise Exception("If model use reverse channel"
                                "reverse augmentation must be performed")
                
    def dump(self):
        self.to_json(self.experiment_config_path)
        
    def to_dict(self) -> dict:
        dt = asdict(self)
        return dt
    
    def to_json(self, path: str | Path):
        dt = self.to_dict()
        with open(path, 'w') as out:
            json.dump(dt, out, indent=4)
            
    @classmethod
    def from_dict(cls, dt: dict) -> 'Experiment':
        model_cfg = LegNetConfig(**dt['model_cfg'])
        train_scheme_cfg = TrainingSchemeConfig(**dt['train_scheme_cfg'])
        data_cfg = DreamDataModuleConfig(**dt['data_cfg'])
        run_cfg = RunConfig(**dt['run_cfg'])
        global_seed = int(dt['global_seed'])
        train_log_step = int(dt['train_log_step'])
        set_medium_float32_precision = bool(dt['set_medium_float32_precision'])
        
        return cls(model_cfg=model_cfg,
                   train_scheme_cfg=train_scheme_cfg, 
                   data_cfg=data_cfg, 
                   run_cfg=run_cfg,
                   global_seed=global_seed,
                   set_medium_float32_precision=set_medium_float32_precision,
                   train_log_step=train_log_step)
        
    @classmethod
    def from_json(cls, path: Path | str) -> 'Experiment':
        with open(path, 'r') as inp:
            dt = json.load(inp)
        return cls.from_dict(dt)
    
    