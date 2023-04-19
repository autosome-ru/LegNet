import argparse
from random import choice

from dream_lightning_wrapper import LegNetConfig, TrainingSchemeConfig
from dream_datamodule import DreamDataModuleConfig
from dream_run import RunConfig
from experiment import Experiment


parser = argparse.ArgumentParser(description="Script to run experiments for LegNet-DREAM2022 paper", 
                                 allow_abbrev=False)


parser.add_argument('--seed',
                    required=True, 
                    type=int)
parser.add_argument('--train_log_step',
                    default=1000,
                    type=int)

model_args = parser.add_argument_group('model architecture', 
                                       'architecture parameters')
model_args.add_argument("--use_single_channel", 
                        action="store_true")
model_args.add_argument("--use_reverse_channel",
                        action="store_true")
model_args.add_argument("--kernel_size", 
                        required=True, 
                        type=int)
model_args.add_argument("--filter_per_group", 
                        required=True, 
                        type=int)
model_args.add_argument("--res_block_type",
                        choices=['concat', 
                                 'add',
                                 'none'], 
                        type=str, 
                        required=True)
model_args.add_argument("--se_type",
                        choices=['none', 
                                 'simple', 
                                 'complex'], 
                        type=str, 
                        required=True)
model_args.add_argument("--inner_dim_calculation",
                        choices=['in', 
                                 'out'], 
                        type=str, 
                        required=True)
model_args.add_argument("--final_activation",
                        choices=['silu', 
                                 'none'], 
                        type=str, 
                        required=True)
model_args.add_argument("--blocks", 
                    required=True,
                    nargs="+", 
                    type=int, 
                    help="number of channels for EffNet-like blocks")

model_args.add_argument("--resize_factor", default=4, type=int)
model_args.add_argument("--se_reduction", default=4, type=int)
model_args.add_argument("--activation",
                        choices=['silu'], 
                        type=str, 
                        default='silu')


train_scheme_args = parser.add_argument_group('train_scheme', 
                                       'training scheme parameters')
train_scheme_args.add_argument("--train_mode", 
                               choices=['classification',
                                        'regression'],
                               type=str,
                               required=True)
train_scheme_args.add_argument("--optimizer", 
                               choices=['adamw',
                                        'lion'],
                               type=str,
                               required=True)
train_scheme_args.add_argument("--scheduler", 
                               choices=['onecycle',
                                        'reduceonplateau'],
                               type=str,
                               required=True)
train_scheme_args.add_argument("--lr", 
                               type=float,
                               required=True)
train_scheme_args.add_argument("--wd", 
                               type=float,
                               required=True)

data_args = parser.add_argument_group('data', 
                                      'data preparation args')
data_args.add_argument("--split_type", 
                       choices=['fulltrain', 'trainvalid'],
                       required=True)
data_args.add_argument("--dataset_path", 
                       type=str,
                       required=True)
data_args.add_argument('--reverse_augment',
                       action="store_true")

data_args.add_argument('--seqsize',
                       type=int,
                       default=150)
data_args.add_argument('--shift',
                       type=float,
                       default=0.5)
data_args.add_argument('--scale',
                       type=float,
                       default=0.5)
data_args.add_argument('--train_batch_size',
                       type=int,
                       default=1024)
data_args.add_argument('--valid_batch_size',
                       type=int,
                       default=4096)
data_args.add_argument('--num_workers',
                       type=int,
                       default=16)
model_args.add_argument("--valid_folds", 
                    nargs="*", 
                    type=int, 
                    default=[])
data_args.add_argument('--fold_hash_seed',
                       type=int,
                       default=42)

run_args = parser.add_argument_group('run', 
                                      'Run params')
run_args.add_argument("--model_dir",
                      required=True,
                      type=str)
run_args.add_argument("--max_steps",
                      default=80000,
                      type=int)
run_args.add_argument("--val_check_interval",
                      default=1000,
                      type=int)
run_args.add_argument("--accelerator",
                      default="gpu",
                      type=str)
run_args.add_argument("--device",
                      default=0,
                      type=int)
run_args.add_argument("--precision",
                      default="bf16-mixed",
                      type=str)
run_args.add_argument("--train_log_every_n_steps",
                      default=50,
                      type=int)
run_args.add_argument("--fmt_precision",
                      default=4,
                      type=int)
run_args.add_argument("--save_k_best",
                      default=1,
                      type=int)
run_args.add_argument("--metric",
                      choices=['val_pearson', 
                               'val_loss', 
                               'val_pearson'],
                      default="val_pearson",
                      type=str)
run_args.add_argument("--direction",
                      choices=['min', 'max'],
                      default="max",
                      type=str)


args = parser.parse_args()
 
experiment = Experiment(
    model_cfg = LegNetConfig(
        use_single_channel=args.use_single_channel,
        use_reverse_channel=args.use_reverse_channel,
        block_sizes=args.blocks,
        ks=args.kernel_size,
        resize_factor=args.resize_factor,
        activation=args.activation,
        final_activation=args.final_activation,
        filter_per_group=args.filter_per_group,
        se_reduction=args.se_reduction,
        res_block_type=args.res_block_type,
        se_type=args.se_type,
        inner_dim_calculation=args.inner_dim_calculation),
    train_scheme_cfg = TrainingSchemeConfig(
        train_mode=args.train_mode, 
        optimizer_type=args.optimizer, 
        scheduler=args.scheduler,
        lr=args.lr,
        weight_decay=args.wd),
    data_cfg = DreamDataModuleConfig(
        split_type=args.split_type, 
        dataset_path=args.dataset_path,
        add_single_column=args.use_single_channel,
        reverse_augment=args.reverse_augment,
        seqsize=args.seqsize,
        shift=args.shift,
        scale=args.scale,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers, 
        valid_folds=args.valid_folds,
        fold_hash_seed=args.fold_hash_seed),
    run_cfg=RunConfig(
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        accelerator=args.accelerator,
        devices=[args.device],
        precision=args.precision,
        model_dir=args.model_dir,
        train_log_every_n_steps=args.train_log_every_n_steps,
        fmt_precision=args.fmt_precision,
        k_best=args.save_k_best,
        metric=args.metric,
        direction=args.direction),
    global_seed=args.seed,
    set_medium_float32_precision=True,
    train_log_step=args.train_log_step)


experiment.run()