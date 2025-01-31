from mordred import Calculator, descriptors
from rdkit import Chem
import pandas as pd
import os

'''
This descriptor needs to be performed with numpy version < 1.20. Note to self: Activate environment 'mordred' before calc this descriptors
'''
def get_mordred(data):  
    calc = Calculator(descriptors, ignore_3D=False)
    desc_list = []
    smi_idx = []
    
    for idx in range(len(data)):
        mol_file = f'C:/Users/Ng Ping Boon/Desktop/FYP/3D Conformers/xtb/mol/xtbopt_{idx}.mol'
        if os.path.exists(mol_file):
            mol = Chem.MolFromMolFile(mol_file, removeHs=False)
            desc = calc(mol)
            desc_list.append(desc)
            smi_idx.append(idx)
    df_desc = pd.DataFrame(desc_list, columns=[str(key) for key in desc.keys()])
    var_per_col = list(df_desc.var())
    zero_var_desc = []
    for idx, var in enumerate(var_per_col):
        if var == 0:
            zero_var_desc.append(idx)
    df_desc = df_desc.drop(columns=df_desc.columns[zero_var_desc])
    
    smi = data['SMILES'][smi_idx]
    smi.reset_index(drop=True, inplace=True)
    df_desc['SMILES'] = smi
    return df_desc