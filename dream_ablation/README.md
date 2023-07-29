This folder contains the code for the ablation study in the Penzar et al. (doi: 10.1093/bioinformatics/btad457). 

[Training](train.py) and [testing](test.py) scripts allow to vary every parameter adjusted in the ablation study. Final optimized model can be trained via [train.sh](train.sh).

For detailed description of all the changes introduced to the original approach please refer to the [tutorial](../tutorial/demo_notebook_optimized.ipynb).

## train.py arguments

| Parameter                 | Default       | Type/Action | Description   |	
| :------------------------ |:-------------:| :-------------:| :-------------|
| --seed 	       |	-           | int |random seed
| --train_log_step          | 1000           | int |frequency of steps with which to save snapshots during the train epoch
| <td colspan=3>model architecture parameters
| --use_single_channel 	       |	- | store_true	            |if True, singleton augmentation will be performed
| --use_reverse_channel  		       | - | store_true	           | if True, reverse augmentation will be performed
| --kernel_size 		           | -           | int  | kernel size of convolutional layers
| --filter_per_group 	        | -       | int   | maximum distance allowed between DMRs to merge 
| --res_block_type	         | -          | str: concat / add / none | type of residual block, see [tutorial](../tutorial/demo_notebook_optimized.ipynb)
| --se_type          | -   | str: none / simple / complex     | type of SE block, see [tutorial](../tutorial/demo_notebook_optimized.ipynb)
| --inner_dim_calculation       | -  | str: in / out | mechanism of dimention calculation inside EfficientNet-like block, see [tutorial](../tutorial/demo_notebook_optimized.ipynb)
| --final_activation    | -   | str: silu / none   | presence of the activation after core blocks, see [tutorial](../tutorial/demo_notebook_optimized.ipynb)
| --blocks			             | -  | int, + 	           | number of channels for EffNet-like blocks
| --resize_factor			     | 4  | int        | resize factor used in a high-dimensional middle layer of an EffNet-like block
| --se_reduction  | 4	 | int     	     | reduction number used in SELayer 
| --activation		    | silu  | str: silu     | activation function
| <td colspan=3>training scheme parameters
| --train_mode		      | -  | str: classification / regression | train mode
| --optimizer             | -  | str: adamw / lion | optimizer choice, see [tutorial](../tutorial/demo_notebook_optimized.ipynb)
| --scheduler             | - | str: onecycle / reduceonplateau | scheduler choice, see [tutorial](../tutorial/demo_notebook_optimized.ipynb)
| --lr                    | - | float | learning rate value, will be divided by lr_div if scheduler is set to "onecycle"
| --wd                    | - | float | weight decay coefficient
| <td colspan=3>data preparation parameters
| --split_type            | - | str: fulltrain / trainvalid | train on the whole training set / split the training data to train and validation
| --dataset_path          | - | str | path to training dataset
| --reverse_augment       | - | store_true | if True, reverse augmentation on training data will be performed
| --seqsize               | 150 | int | length to pad the input sequence to
| --shift                 |  0.5 |  float | assumed sd of real expression normal distribution
| --scale                 | 0.5  | float  | assumed scale of real expression normal distribution
| --train_batch_size      | 1024 | int    | train batch size
| --valid_batch_size      | 4096 | int    | validation batch size
| --num_workers           | 16   | int    | number of workers
| --valid_folds           | []   | int, * | number of validation folds if --split_type is trainvalid
| --fold_hash_seed        | 42   | int    | seed used in randomly splitting to validation folds
| <td colspan=3>run parameters
| --model_dir             | - | str | folder that will contatin all outputs
| --max_steps             | 80000 | int | max_steps parameter for pl.Trainer
| --val_check_interval    | 1000  | int | val_check_interval parameter for pl.Trainer
| --accelerator           | gpu   | str | accelerator type used for training
| --device                | 0     | int | device number used for training
| --precision             | bf16-mixed | str | arithmetic for pytorch model training
| --train_log_every_n_steps | 50 | int | train_log_every_n_steps parameter for pl.Trainer
| --fmt_precision         | 4 | int | fmt_precision parameter for IterativeCheckpointCallback
| --save_k_best           | 1 | int | k_best parameter for IterativeCheckpointCallback
| --metric                | val_pearson | str: val_pearson / val_loss | monitor parameter for IterativeCheckpointCallback
| --direction             | max  | str: min / max  | direction parameter for IterativeCheckpointCallback