from rdkit import Chem
import functools
import numpy as np
import pandas as pd

def unit_conversion(log_raw_data): # Scale to Log(Papp x 10^6)
    raw_data = 10**log_raw_data
    scaled_log = np.log10(raw_data*(10**6))
    return scaled_log

def subtract_dict_values(dictionary):
    return functools.reduce(lambda x, y: x - y, dictionary.values())

def process_smiles(duplicated_dict, idx, df, finalized_df, smi): # Processing logPapp values according to conditions stated in 1_1_preprocessing.py
    if idx < len(df) - 1:
        while smi == df['SMILES'].iloc[idx + 1]:
            duplicated_dict[idx + 1] = df['logPapp'].iloc[idx + 1]
            idx += 1
            if idx >= len(df) - 1:
                break
        if subtract_dict_values(duplicated_dict) == 0:
            finalized_df.loc[len(finalized_df)] = [smi, df['logPapp'].iloc[idx]]
        else:
            min_value = min(duplicated_dict.values())
            max_value = max(duplicated_dict.values())
            if (max_value - min_value) <= 0.1:
                average = sum(duplicated_dict.values()) / len(duplicated_dict.values())
                finalized_df.loc[len(finalized_df)] = [smi, average]
            else:
                pass
        return finalized_df

def handle_duplicates(df):
    finalized_df = pd.DataFrame(columns=['SMILES', 'logPapp'])
    dummy_smi = df['SMILES'].iloc[0]
    for idx, smi in enumerate(df['SMILES']):
        if idx == 0:
            duplicated_dict = {}
            duplicated_dict[idx] = df['logPapp'].iloc[idx]
            finalized_df = process_smiles(duplicated_dict, idx, df, finalized_df, smi)
        else:
            if dummy_smi != smi:
                dummy_smi = smi
                duplicated_dict = {}
                duplicated_dict[idx] = df['logPapp'].iloc[idx]
                finalized_df = process_smiles(duplicated_dict, idx, df, finalized_df, smi)
    return finalized_df

def canonicalization(df):
    df.reset_index(drop=True, inplace=True)
    mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    unprocessed_smiles_idx = []
    canon_smiles_idx = []
    canon_smiles = []
    for idx, mol in enumerate(mols):
        try:
            canon_smi = Chem.MolToSmiles(mol)
            canon_smiles_idx.append(idx)
            canon_smiles.append(canon_smi)
        except:
            unprocessed_smiles_idx.append(idx)
    df = df.drop(index = unprocessed_smiles_idx)
    df.reset_index(inplace=True, drop=True)
    df['SMILES'] = canon_smiles
    return df

def identify_duplicates(df):
    df['logPapp'] = df['logPapp'].round(6)
    df_duplicates = df[df.duplicated(subset='SMILES', keep=False)].copy()
    df_duplicates.sort_values(ascending=True, by='SMILES', inplace=True)
    return df_duplicates
