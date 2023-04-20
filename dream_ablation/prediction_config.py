import json

from pathlib import Path
from typing import ClassVar
from dataclasses import dataclass, asdict

from experiment import Experiment


@dataclass
class PredictionConfig:
    seqs_path: str
    pred_col: str
    experiment_path: str 
    model: str
    singleton_mode: str
    devices: list[int]
    out_dir: str
    
    PREDICTION_FILE_NAME: ClassVar[str] = "pred.tsv"
    PRED_CONFIG_FILE_NAME: ClassVar[str] = "pred_params.json"
    
    
    def run(self):
        if Path(self.out_dir).exists():
            raise Exception("Prediction dir already exists")
        Path(self.out_dir).mkdir(parents=True)
        self.dump()
        
        exp = Experiment.from_json(self.experiment_cfg)
        
        if self.model == "best":
            model_path = exp.best_model_path(self.experiment_path)
        elif self.model == "last":
            model_path = exp.last_model_path(self.experiment_path)
        else:
            model_path = self.model
        
        if not Path(model_path).exists():
            raise Exception(f"No such model path: {model_path}")
        
        df = exp.predict(seqs=self.seqs_path,
                model_path=model_path,
                singleton_mode=self.singleton_mode,
                total_pred_col=self.pred_col,
                logdir=self.out_dir)
        df.to_csv(self.prediction_path, 
            sep="\t", 
            index=False)
        
    
    @property
    def experiment_cfg(self):
        return Path(self.experiment_path) / Experiment.CONFIG_PATH_NAME
    
    @property
    def prediction_path(self):
        return Path(self.out_dir) / self.PREDICTION_FILE_NAME
    
    @property
    def prediction_config_path(self):
        return Path(self.out_dir) / self.PRED_CONFIG_FILE_NAME
    
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, dt: dict) -> 'PredictionConfig':
        return cls(**dt)
    
    def to_json(self, path: str | Path):
        with open(path, "w") as out:
            dt = self.to_dict()
            json.dump(dt, 
                      fp=out,
                      indent=4)
    
    @classmethod
    def from_json(cls, path: str | Path) -> 'PredictionConfig':
        with open(path, 'r') as inp:
            dt = json.load(inp)
        return cls.from_dict(dt)
    
    def dump(self):
        self.to_json(self.prediction_config_path)
