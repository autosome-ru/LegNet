import pandas as pd

mediums = ['complex', 'defined']

for medium in mediums:
    df = pd.read_csv(f'../../../data/output/{medium}_medium_drift_res.tsv', sep='\t')
    
    evol_prediction = pd.read_csv(f'../../../data/output/drift_original_{medium}.txt', header=None, sep='\t')
    
    df['evol_predicted'] = evol_prediction[1]
    df['delta_predicted'] = df['evol_predicted'] - df['original_predicted']

    authors_drift_delta = pd.read_csv(f'../../../data/input/{medium}_medium_Drift_delta.csv')
    authors_drift_evol = pd.read_csv(f'../../../data/input/{medium}_medium_Drift.csv')
    
    df['evol_predicted_auth'] = authors_drift_evol['Predicted Expression']
    df['delta_predicted_auth'] = authors_drift_delta['Predicted Expression']
    df['original_predicted_auth'] = df.evol_predicted_auth - df.delta_predicted_auth
    
    df.to_csv(f'../../../data/output/{medium}_medium_drift_res.tsv', sep='\t', index=None)