python3 ../../legnet/test.py --target ../../data/input/defined_medium_Native_expression.txt --output ../../data/output/defined_Native.txt --seed 42 --valid_batch_size 1024 --valid_workers 8 --seqsize 150 --temp .TEMPDIR --use_single_channel --singleton_definition integer --gpu 1 --ks 7 --blocks 256 128 128 64 64 64 64 --resize_factor 4 --se_reduction 4 --final_ch 18 --delimiter tab --output_format tsv --model ../../models/complex/model_150.pth
python3 ../../legnet/test.py --target ../../data/input/defined_medium_Drift_expression.txt --output ../../data/output/defined_Drift.txt --seed 42 --valid_batch_size 1024 --valid_workers 8 --seqsize 150 --temp .TEMPDIR --use_single_channel --singleton_definition integer --gpu 1 --ks 7 --blocks 256 128 128 64 64 64 64 --resize_factor 4 --se_reduction 4 --final_ch 18 --delimiter tab --output_format tsv --model ../../models/complex/model_150.pth