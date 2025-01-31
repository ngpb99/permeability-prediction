from utils.lgbm_mordred_utils import get_mordred
import pandas as pd

'''
Run with 'mordred' environment!!
'''
data = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/3D_optimizable_dataset.csv')
mordred_desc = get_mordred(data)
obj_cols = mordred_desc.select_dtypes(include=['object']).columns
unchanged_cols = []
for col in obj_cols:
    try:
        mordred_desc[col] = mordred_desc[col].astype('float')
    except ValueError:
        unchanged_cols.append(col)
        
# ^ All changed to float, which means no strings. They were 'object' dtypes because they contain NA. However, isna() cannot identify NA from objects

col_na = mordred_desc.isna().sum(axis=0)
cols_to_drop = col_na[col_na > 50].index # Drop cols with more than 50 NA
mordred_desc.drop(columns=cols_to_drop, inplace=True)
mordred_desc.dropna(axis=0, inplace=True) # Drop rows with NA
mordred_desc.reset_index(inplace=True, drop=True)
bool_cols = mordred_desc.select_dtypes(include=['bool']).columns
mordred_desc.loc[:, bool_cols] = mordred_desc.loc[:, bool_cols].astype(int)

# ^ Above was to make sure not too mch info was dropped. 
# The cols_to_drop was to ensure that any columns with more than 50 NA (arbitrary), we drop it.
# Some cols contain only a few entries with NA. Dropping those entirely would result in loss of valuable info.
# This function was to trade-off between loss of molecules (row-wise), and descriptor info (column-wise)
# In total, we lost 77 molecules.

train = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/train_data.csv')
test = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/test_data.csv')
train_mordred = pd.merge(train, mordred_desc, on='SMILES', how='inner')
test_mordred = pd.merge(test, mordred_desc, on='SMILES', how='inner')

train_mordred.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_train.csv', index=False)
test_mordred.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_test.csv', index=False)