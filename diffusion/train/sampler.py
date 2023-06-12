import numpy as np 
import pandas as pd
import os



def load_full(path_from, start=0, end=17, length=None, N_contain=False):
    df = pd.read_csv(path_from, sep = "\t", header=None)
    df = df.drop(2, axis=1)
    df.columns = ['seq', 'expression']
    
    if (start >= df["expression"].min()) or (end <= df["expression"].max()):
        # Filtering by expression
        df = df[(df["expression"] >= start) & (df["expression"] <= end)]

    if not N_contain:
        if length is not None:
            df = df[(df["seq"].str.len() == length) & (~df["seq"].str.contains('N'))]
        else:
            df = df[(~df["seq"].str.contains('N'))]
    else:
        if length is None:
            df = df[(df["seq"].str.len() == length)]
        else:
            df = df[(~df["seq"].str.contains('N'))]
    df.reset_index(drop=True, inplace=True)
    return df


