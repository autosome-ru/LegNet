from dataclasses import dataclass, asdict 

import json
import torch

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
    
    def run(self):
        self.dump()
        set_global_seed(self.global_seed)
        if self.set_medium_float32_precision:
            torch.set_float32_matmul_precision('medium') 
            
        model = DreamModel(model_cfg=self.model_cfg,
                   train_cfg=self.train_scheme_cfg,
                   train_log_step=self.train_log_step,
                   for_lr_finder=False)
        data = self.data_cfg.get_datamodule()
        trainer = self.run_cfg.get_trainer(validation_exists=self.data_cfg.validation_exists,
                                           save_every_n_train_steps=self.train_log_step)
        trainer.fit(model, data)

    @property
    def root_dir(self) -> Path:
        return Path(self.run_cfg.model_dir)
    
   
    def check_params(self): 
        if self.root_dir.exists():
            raise Exception(f"Model dir already exists: {self.root_dir}")
        if not self.data_cfg.reverse_augment:
            if self.model_cfg.use_reverse_channel:
                raise Exception("If model use reverse channel"
                                "reverse augmentation must be performed")
                
    def __post_init__(self):
        self.check_params()
        self.root_dir.mkdir(parents=True)
        
    def dump(self):
        p = self.root_dir / 'params.json'
        self.to_json(p)
        
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
    
    