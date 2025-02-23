import lightning.pytorch as pl 
from pathlib import Path
from dataclasses import dataclass
from iterative_checkpoint_callback import IterativeCheckpointCallback
@dataclass
class RunConfig:
    max_steps: int
    val_check_interval: int
    accelerator: str
    devices: list[int]
    precision: str 
    model_dir: str
    train_log_every_n_steps: int = 50
    fmt_precision: int = 4
    k_best: int = 1
    metric: str = "val_pearson"
    direction: str = "max"
    
    def get_trainer(self, 
                    validation_exists: bool, 
                    save_every_n_train_steps: int):
        callbacks = []
    
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",# type: ignore
                                                      log_momentum=True) 
        callbacks.append(lr_monitor)

        if validation_exists:
            best_checkpoint_callback = IterativeCheckpointCallback(monitor=self.metric, 
                                                                   k_best=self.k_best, 
                                                                   direction=self.direction,
                                                                   fmt_precision=self.fmt_precision)
            callbacks.append(best_checkpoint_callback)
        
        last_checkpoint_callback = pl.callbacks.ModelCheckpoint(   #type: ignore
            save_top_k=1,
            monitor="step",
            mode="max",
            filename="last_model-{step}",
            save_on_train_epoch_end=False,
            every_n_train_steps=save_every_n_train_steps,
        )
        callbacks.append(last_checkpoint_callback)

        return pl.Trainer(
            max_steps=self.max_steps,
            val_check_interval=self.val_check_interval,
            check_val_every_n_epoch=None,
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision, # type: ignore 
            default_root_dir=self.model_dir,
            max_epochs=-1,
            callbacks=callbacks,
            log_every_n_steps=self.train_log_every_n_steps,
            enable_checkpointing=True,
        )
        
    def get_predict_runner(self, logdir: str | Path):
        runner = pl.Trainer(accelerator=self.accelerator,
                             devices=self.devices,
                             default_root_dir=logdir)
        return runner