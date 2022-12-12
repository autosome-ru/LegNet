from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

mediums = ['complex', 'defined']

evol_df = pd.read_csv(f'../../../data/input/defined_medium_evol_seqs.txt', sep='\t')
evol_df_complex = pd.read_csv(f'../../../data/input/complex_medium_evol_seqs.txt', sep='\t')

for medium in mediums:
    df_measured = pd.read_csv(f'../../../data/input/{medium}_medium_Drift_expression.txt', sep='\t', header=None)
    authors_delta = pd.read_csv(f'../../../data/input/{medium}_medium_Drift_delta.csv')
    
    df_pred = pd.read_csv(f'../../../data/output/{medium}_Drift.txt', sep='\t', header=None)
    
    seqs = df_measured[0]
    
    original_seqs = []
    n_muts = []
    original_measured, original_predicted = [], []
    evol_measured = []
    delta_measured = []


    for seq in tqdm(seqs):
        needed_subset = evol_df[evol_df.seq110 == seq]
        original, mutations = needed_subset.origID.iloc[0].split('_')
        n_mut = len(mutations.split('.'))
        n_muts.append(n_mut)

        expression_evol = float(evol_df[evol_df.origID == original].ExpressionLevel)

        if medium == 'complex':
            seq_evol = evol_df[evol_df.origID == original].iloc[0,2]
            expression_evol = float(evol_df_complex[evol_df_complex.seq110 == seq_evol].NBT_S288CdU_YPD.iloc[0])

        original_seq = evol_df[evol_df.origID == original].seq110.iloc[0]
        expression_measured = float(df_measured[df_measured[0] == seq][1])
        expression_delta = float(authors_delta[authors_delta.sequence == seq]['Measured Expression'])
        expression_predicted = float(df_pred[df_pred[0] == seq][1])

        original_seqs.append(original_seq)
        original_measured.append(expression_measured)
        original_predicted.append(expression_predicted)
        evol_measured.append(expression_evol)
        delta_measured.append(expression_delta)
        
    df = pd.DataFrame({'seq': seqs, 
              'original_seq': original_seqs, 
              'n_mut': n_muts,   
              'original_measured': original_measured,
              'evol_measured': evol_measured,
              'delta_measured': delta_measured,
              'original_predicted': original_predicted, 
             })

    df.to_csv(f'../../../data/output/{medium}_medium_drift_res.tsv', sep='\t', index=None)