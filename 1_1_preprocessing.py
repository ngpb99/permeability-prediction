import pandas as pd
import numpy as np
from utils.preprocessing_utils import unit_conversion, canonicalization, handle_duplicates, identify_duplicates

'''
1) initial processing of data: identify duplicates, canonicalizing smiles, scaling logPapp
'''
df1 = pd.read_excel('./Caco-2 Permeability Dataset/ci0c00568_si_002.xls') # Already scaled
df1 = df1[['SMILES', 'LogPapp Value']]
df1.rename(columns={'LogPapp Value':'logPapp'}, inplace=True)
df1.dropna(axis=0, inplace=True)
canonicalized_df1 = canonicalization(df1)
df1_duplicates = identify_duplicates(canonicalized_df1)

df2 = pd.read_excel('./Caco-2 Permeability Dataset/ci5b00642_si_001.xlsx')
df2 = df2[['smi', 'logPapp']]
df2.rename(columns={'smi':'SMILES'}, inplace=True)
df2['logPapp'] = pd.to_numeric(df2['logPapp'],errors = 'coerce')
df2['logPapp'] = unit_conversion(df2['logPapp'])
df2.dropna(axis=0, inplace=True)
canonicalized_df2 = canonicalization(df2)
df2_duplicates = identify_duplicates(canonicalized_df2)

df3 = pd.read_excel('./Caco-2 Permeability Dataset/LogPapp_Data.xlsx')
df3 = df3[['SMILES', 'logPapp']]
df3['logPapp'] = unit_conversion(df3['logPapp'])
df3.dropna(axis=0, inplace=True)
canonicalized_df3 = canonicalization(df3)
df3_duplicates = identify_duplicates(canonicalized_df3)

'''
2) Dealing with duplicates in df1, df2 and df3.

Conditions for duplicates:
    1) Same LogPapp, keep first entry
    2) Different LogPapp with small difference (<= 0.1), take average
    3) Different LogPapp with big difference (> 0.1), remove duplicates
'''
processed_df1_duplicates = handle_duplicates(df1_duplicates)
processed_df2_duplicates = handle_duplicates(df2_duplicates)
processed_df3_duplicates = handle_duplicates(df3_duplicates)
canonicalized_df1.drop_duplicates(inplace=True, keep=False, subset='SMILES')
canonicalized_df2.drop_duplicates(inplace=True, keep=False, subset='SMILES')
canonicalized_df3.drop_duplicates(inplace=True, keep=False, subset='SMILES')
cleaned_df1 = pd.concat([canonicalized_df1, processed_df1_duplicates], axis=0)
cleaned_df2 = pd.concat([canonicalized_df2, processed_df2_duplicates], axis=0)
cleaned_df3 = pd.concat([canonicalized_df3, processed_df3_duplicates], axis=0)
cleaned_df1.reset_index(inplace=True, drop=True)
cleaned_df2.reset_index(inplace=True, drop=True)
cleaned_df3.reset_index(inplace=True, drop=True)

'''
3) Dealing with duplicates within all 3 datasets. 
   Better to calculate the average of molecule A's values within each dataset first, and then average these dataset-level averages across the three datasets. 
   This ensures that the final average is not disproportionately influenced by datasets with more occurrences of molecule A.
'''
merged = pd.concat([cleaned_df1, cleaned_df2, cleaned_df3], axis=0)
merged_duplicates = identify_duplicates(merged)
processed_merged_duplicates = handle_duplicates(merged_duplicates)
merged.drop_duplicates(inplace=True, keep=False, subset='SMILES')
cleaned_merged = pd.concat([merged, processed_merged_duplicates], axis=0)
cleaned_merged.reset_index(inplace=True, drop=True)
cleaned_merged['priority'] = 1

'''
4) Dealing with duplicates in OCHEM database. 
   Assume symmetry for bidirection, as majority of data do not include directional information.
'''
df4 = pd.DataFrame(columns=['SMILES', 'logPapp'])
for i in range(1, 10):
    data = pd.read_csv(f'./Caco-2 Permeability Dataset/papp_dataset_{i}.csv')
    data = data[['SMILES', 'Papp(Caco-2) {measured, converted}']]
    data['Papp(Caco-2) {measured, converted}'] = np.log10(data['Papp(Caco-2) {measured, converted}'])
    data.rename(columns={'Papp(Caco-2) {measured, converted}':'logPapp'}, inplace=True)
    df4 = pd.concat([df4, data], axis=0)
df4.dropna(axis=0, inplace=True)
df4 = df4.drop(df4[df4['logPapp'] == -np.inf].index) # infinity values cause by Papp = 0, log(0) = error.
canonicalized_df4 = canonicalization(df4)
df4_duplicates = identify_duplicates(canonicalized_df4)

processed_df4_duplicates = handle_duplicates(df4_duplicates)
canonicalized_df4.drop_duplicates(inplace=True, keep=False, subset='SMILES')
cleaned_df4 = pd.concat([canonicalized_df4, processed_df4_duplicates], axis=0)
cleaned_df4.reset_index(inplace=True, drop=True)
cleaned_df4['priority'] = 2

'''
5) Removing entries in OCHEM that is found in merged df1, df2, and df3 (cleaned_merged). 
   Rationale: existing computational work has been performed with them and obtained good results, higher confidence.
   Filter conditions in accordance to 10.1039/d0ra08209k and 10.1021/acs.jcim.5b00642. Only 21 entries lost.
'''
merged_df = pd.concat([cleaned_merged, cleaned_df4], axis=0)
merged_df.sort_values(ascending=True, by='priority', inplace=True)
merged_df.drop_duplicates(inplace=True, keep='first', subset='SMILES')
merged_df.reset_index(inplace=True, drop=True)
filtered_df = merged_df[(merged_df['logPapp'] < 2.5) & (merged_df['logPapp'] > -2)]
filtered_df.to_csv('./Caco-2 Permeability Dataset/LogPapp_cleaned_dataset.csv', index=False)

cleaned_df1.to_csv('./Caco-2 Permeability Dataset/Wang_et_al2020.csv', index=False)
cleaned_df2.to_csv('./Caco-2 Permeability Dataset/Wang_et_al2016.csv', index=False)
cleaned_df3.to_csv('./Caco-2 Permeability Dataset/Wang_and_Chen2020.csv', index=False)

