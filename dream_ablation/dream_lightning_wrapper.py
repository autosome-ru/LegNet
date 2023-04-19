
import torch
import lightning.pytorch  as pl
from typing import Type 
from torch import nn 
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

from dataclasses import dataclass, asdict
from lion import Lion
from legnet import LegNet

@dataclass
class LegNetConfig: 
    use_single_channel: bool 
    use_reverse_channel: bool
    block_sizes: list[int]
    ks: int 
    resize_factor: int
    activation: str
    final_activation: str
    filter_per_group: int
    se_reduction: int
    res_block_type: str
    se_type: str
    inner_dim_calculation: str
    
    def _str2activation(self, act: str) -> Type[nn.Module]:
        if act == 'silu':
            return nn.SiLU
        elif act == "none":
            return nn.Identity
        else:
            raise Exception(f"Wrong activation type: {act}")    
        
    def get_activation(self) -> Type[nn.Module]:
        return self._str2activation(self.activation)
    
    def get_final_activation(self) -> Type[nn.Module]:
        return self._str2activation(self.final_activation)
        
    def get_model(self) -> nn.Module:
        model = LegNet(use_single_channel=self.use_single_channel,
                       use_reverse_channel=self.use_reverse_channel,
                       block_sizes=self.block_sizes, 
                       ks=self.ks, 
                       resize_factor=self.resize_factor,
                       activation = self.get_activation(),
                       final_activation= self.get_final_activation(),
                       filter_per_group=self.filter_per_group,
                       se_reduction=self.se_reduction,
                       res_block_type=self.res_block_type,
                       se_type=self.se_type,
                       inner_dim_calculation=self.inner_dim_calculation)
        
        return model


@dataclass
class TrainingSchemeConfig:
    train_mode: str
    optimizer_type: str
    lr: float
    weight_decay: float
    scheduler: str 
    cycle_momentum: bool = False
    lr_div: float = 25.0 # division for onecyclelr
    pct_start: float = 0.3
    three_phase: bool = False
    
    def get_optimizer_class(self):
        if self.optimizer_type == "adamw":
            return torch.optim.AdamW
        elif self.optimizer_type == "lion":
            return Lion
        raise Exception(f"Wrong optimizer_type: {self.optimizer_type}")
    
    @property
    def optim_lr(self):
        if self.scheduler == "onecycle":
            return self.lr / self.lr_div
        else:
            return self.lr 
    
    def get_optimizer(self, model: nn.Module):  
        if self.optimizer_type == "adamw":
             
            return torch.optim.AdamW(model.parameters(), 
                                     lr=self.optim_lr,
                                     weight_decay=self.weight_decay)
        elif self.optimizer_type == "lion":
            return Lion(model.parameters(),
                        lr=self.optim_lr,
                        weight_decay=self.weight_decay)
        else:
            raise Exception(f"Wrong optimizer type: {self.optimizer_type}")
        
    def _get_onecyclelr_scheduler(self, optim: torch.optim.Optimizer, trainer: pl.Trainer) -> dict:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, 
                                                           max_lr=self.lr,
                                                           three_phase=self.three_phase, 
                                                           total_steps=trainer.estimated_stepping_batches, # type: ignore 
                                                           pct_start=self.pct_start,
                                                           cycle_momentum =self.cycle_momentum)
        lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
               "frequency": 1,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "cycle_lr"
        }
        return lr_scheduler_config
    
    def _get_reduceonplateau_scheduler(self, optim: torch.optim.Optimizer) -> dict:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode='max',
            patience=5,
            factor=0.1)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "reduce",
            "monitor": 'val_pearson'
        }
        
        return lr_scheduler_config
        
    def get_scheduler(self, optim: torch.optim.Optimizer, trainer: pl.Trainer) -> dict:
        if self.scheduler == "onecycle":
            return self._get_onecyclelr_scheduler(optim, trainer)
        elif self.scheduler == "reduceonplateau":
            return self._get_reduceonplateau_scheduler(optim)
        else:
            raise Exception(f"Wrong scheduler type: {self.scheduler}")
       

class DreamModel(pl.LightningModule):
    def __init__(self, 
                 model_cfg: LegNetConfig, 
                 train_cfg: TrainingSchemeConfig,
                 train_log_step: int=1000,
                 for_lr_finder: bool = False,):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        
        self.model = self.model_cfg.get_model()
             
        self.classification_loss = nn.KLDivLoss( reduction= "batchmean")
        self.regression_loss = nn.MSELoss()
        self.for_lr_finder = for_lr_finder
        
        self.train_pearson = PearsonCorrCoef()
        self.train_spearman = SpearmanCorrCoef()
        self.val_pearson = PearsonCorrCoef()
        self.val_spearman = SpearmanCorrCoef()
        self.train_log_step = train_log_step 


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        X, y_probs, y = batch
        y_probs = y_probs.float()
        y = y.float()
        
        if self.train_cfg.train_mode == "classification":
            logprobs, y_hat = self.model(X)
            loss = self.classification_loss(logprobs, y_probs)
        elif self.train_cfg.train_mode == "regression":
            preds = self.model(X)
            if isinstance(preds, tuple):
                y_hat = preds[-1]
            else:
                y_hat = preds
            
            loss = self.regression_loss(y_hat, y)
        else:
            raise Exception("Internal error") 
              
        #with torch.inference_mode():
        self.log("train_loss", loss, prog_bar=True,  on_step=True,  logger=True)  
            
        if self.train_cfg.train_mode == "classification":
            mse = self.regression_loss(y_hat, y)
            self.log("train_mse", mse, prog_bar=True,  on_step=True, on_epoch=False,  logger=True)
        self.train_pearson(y_hat, y)
        self.train_spearman(y_hat, y)
            
        if self.global_step > 0 and self.global_step % self.train_log_step == 0:
            self.log('train_pearson', self.train_pearson.compute(), on_step=True, on_epoch=False,  logger=True, prog_bar=True)
            self.log('train_spearman', self.train_spearman.compute(), on_step=True, on_epoch=False,  logger=True, prog_bar=True)
            self.train_pearson.reset()
            self.train_spearman.reset()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.float()
        preds = self.model(X)
        if isinstance(preds, tuple):
            y_hat = preds[-1]
        else:
            y_hat = preds
        
        loss = self.regression_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True,  on_step=False, on_epoch=True,  logger=True)
        self.val_pearson(y_hat, y)
        self.val_spearman(y_hat, y)
        self.log('val_pearson', self.val_pearson, 
                 on_step=False,
                 on_epoch=True,
                 logger=True)
        self.log('val_spearman', self.val_spearman,
                 on_step=False, 
                 on_epoch=True, 
                 logger=True)
    
    
    def configure_optimizers(self):
        optimizer = self.train_cfg.get_optimizer(self)
        if self.for_lr_finder:
            return optimizer
        lr_scheduler = self.train_cfg.get_scheduler(optimizer, self.trainer)
        return [optimizer], [lr_scheduler]