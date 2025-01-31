from mordred import Calculator, descriptors
import pandas as pd

n_all = Calculator(descriptors, ignore_3D=False).descriptors
n_2D = Calculator(descriptors, ignore_3D=True).descriptors

df_all = pd.DataFrame(n_all)
df_2D = pd.DataFrame(n_2D)
df = pd.concat([df_all, df_2D], axis=0)
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(keep=False, inplace=True)

mordred_train = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_train.csv')
mordred_test = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_test.csv')
df.reset_index(drop=True, inplace=True)
df_str = df[0].astype(str)
desc_3D = list(df_str)
mordred_3D_train = mordred_train.loc[:, desc_3D]
mordred_3D_test = mordred_test.loc[:, desc_3D]

mordred_3D_train['SMILES'] = mordred_train['SMILES']
mordred_3D_test['SMILES'] = mordred_test['SMILES']
mordred_3D_train['logPapp'] = mordred_train['logPapp']
mordred_3D_test['logPapp'] = mordred_test['logPapp']

mordred_3D_train.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_3D_train.csv', index=False)
mordred_3D_test.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_3D_test.csv', index=False)