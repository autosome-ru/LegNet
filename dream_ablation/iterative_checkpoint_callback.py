from dataclasses import dataclass
import lightning.pytorch as pl
import os 
import shutil
import torch

from torch import Tensor

@dataclass
class SavedModel:
    path: str 
    score: float 

class IterativeCheckpointCallback(pl.Callback):
    '''
    Class to fix issue with ModelCheckpoint callback 
    then performing iterative training
    For sure, doesn't implement most functionalities of the ModelCheckpoint
    Issue has been submitted to github 
    '''
    def __init__(self, monitor: str, k_best: int, direction: str, fmt_precision: int = 4):
        self.monitor = monitor
        self.k_best = k_best
        assert self.k_best > 0, f"k_best must be greater then zero"
        self.direction = direction  
        assert self.direction in ('min', 'max'), f"wrong direction: {self.direction}"
        self.cur_bests: list[SavedModel] = []   # should be implemented using heap, but for small number of instances list is OK  
        self.fmt_precision = fmt_precision
    
    def checkpoint_path(self, val: float, step: int) -> str:
        name = f"model_{self.monitor}={val:.0{self.fmt_precision}f}_step={step}.ckpt"
        path = os.path.join(self.dirpath, name)
        return path
    
    def get_step(self, trainer: pl.Trainer) -> int:
        step =  trainer.callback_metrics.get("step")
        step = int(step.int().item()) if isinstance(step, Tensor) else trainer.global_step
        return step
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        monitor_loss = trainer.callback_metrics.get(self.monitor)
        if monitor_loss is None:
            raise Exception(f"Wrong configuration: no such metric in trainer at the validation epoch end: {self.monitor}")
        monitor_loss = float(monitor_loss.item())
        
        if len(self.cur_bests) == self.k_best:
            if self.direction == "max":
                cur_worst = min(self.cur_bests, key=lambda x: x.score)
                if monitor_loss < cur_worst.score: # cur model is the worst
                    return  
            else: # self.direction == "min"
                cur_worst = max(self.cur_bests, key=lambda x: x.score)
                if monitor_loss > cur_worst.score: # cur model is the worst
                    return 
            os.remove(cur_worst.path)
            self.cur_bests.remove(cur_worst)

        step = self.get_step(trainer)
       
        path = self.checkpoint_path(monitor_loss, step)    
        
        saved_model = SavedModel(path, monitor_loss)
        self.cur_bests.append(saved_model)
            
        trainer.save_checkpoint(path)
        
        
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = trainer.strategy.broadcast(dirpath)
        self.dirpath = dirpath
        
    def __resolve_ckpt_dir(self, trainer: "pl.Trainer"):
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:
        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers
        The path gets extended with subdirectory "checkpoints".
        """
        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")
            trainer.strategy.broadcast
        return ckpt_path