Download the needed files to the `data/input` folder before running. 

All `_expression.txt` are preprocessed cooresponing `.csv` files. Preprocessing includes removing the header, index column and the third column (Predicted expression).

### Model training scripts

Script | File location and link 
--- | --- 
[run_defined_base_model_150](run_defined_base_model_150.sh) | [defined_medium_train_expression.txt](https://zenodo.org/record/4436477/files/defined_media_training_data_SC_Ura.txt?download=1)
[run_complex_base_model_300](run_complex_base_model_300.sh) | [complex_medium_train_expression.txt](https://zenodo.org/record/4436477/files/complex_media_training_data_Glu.txt?download=1)

### Model testing scripts

Script | File location and link 
--- | --- 
[run_defined_base_model_150_predict2022](run_defined_base_model_150_predict2022.sh) | [defined_medium_Native_expression.txt](https://zenodo.org/record/4436477/files/Native_defined.csv?download=1)
 &#65279; | [defined_medium_Drift_expression.txt](https://zenodo.org/record/4436477/files/Drift_defined.csv?download=1)
[run_complex_base_model_300_predict2022](run_complex_base_model_300_predict2022.sh) | [complex_medium_Native_expression.txt](https://zenodo.org/record/4436477/files/Native_complex.csv?download=1)
 &#65279; | [complex_medium_Drift_expression.txt](https://zenodo.org/record/4436477/files/Drift_complex.csv?download=1)

### Variant effects-related scripts

Script | File location and link 
--- | --- 
[make_variants_file](variants/make_variants_file.py) | [complex_medium_evol_seqs.txt](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE104nnn/GSE104878/suppl/GSE104878_pTpA_random_design_tiling_etc_YPD_expression.txt.gz)
 &#65279; | [complex_medium_Drift_expression.txt](https://zenodo.org/record/4436477/files/Drift_complex.csv?download=1)
 &#65279; | [complex_medium_Drift_delta.txt](https://zenodo.org/record/4436477/files/Drift_delta_complex.csv?download=1)
 &#65279; | [defined_medium_evol_seqs.txt](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163045/suppl/GSE163045_MolEvol_seq_data_SCUra_only.splitByOrigID.meanEL.all.txt.gz)
 &#65279; | [defined_medium_Drift_expression.txt](https://zenodo.org/record/4436477/files/Drift_defined.csv?download=1)
 &#65279; | [defined_medium_Drift_delta.txt](https://zenodo.org/record/4436477/files/Drift_delta_defined.csv?download=1)
[run_defined_and_complex_predict_variants](variants/run_defined_and_complex_predict_variants.sh) | outputs of [make_variants_file](variants/make_variants_file.py)
[make_full_variants_file](variants/make_full_variants_file.py) | [complex_medium_Drift.csv](https://zenodo.org/record/4436477/files/Drift_complex.csv?download=1)
 &#65279; | [defined_medium_Drift.csv](https://zenodo.org/record/4436477/files/Drift_defined.csv?download=1)

### Figure reproduction

File location and link |
---  | 
[defined_medium_Native.csv](https://zenodo.org/record/4436477/files/Native_defined.csv?download=1)
[complex_medium_Native.csv](https://zenodo.org/record/4436477/files/Native_complex.csv?download=1)
[Native_test_DanQ_model.csv](https://github.com/1edv/evolution/blob/master/manuscript_code/model/results_summary/Native_test_DanQ_model.csv)
[Native_test_DeepAtt_model.csv](https://github.com/1edv/evolution/blob/master/manuscript_code/model/results_summary/Native_test_DeepAtt_model.csv)
[Native_test_DeepSEA_model.csv](https://github.com/1edv/evolution/blob/master/manuscript_code/model/results_summary/Native_test_DeepSEA_model.csv