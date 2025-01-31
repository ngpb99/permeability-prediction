from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import pandas as pd
from umap import UMAP

'''
Calculate Fingerprints of cleaned dataset
'''
data = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/3D_optimizable_dataset.csv')
smiles = data['SMILES'].to_list()
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
cfp_df = pd.DataFrame()

for smile in smiles:
    mol = Chem.MolFromSmiles(smile)
    cfp = mfpgen.GetCountFingerprint(mol)
    cfp = list(cfp)
    cfp_df[smile] = cfp

cfp_df = cfp_df.T
cfp_df.to_csv('./Fingerprints and Dim Reduction Dataset/cfp_cleaned_dataset.csv', index=True)

'''
UMAP reduction (no scaling done as count fingerprints have smae scale, which just counts the no. of substructure occurrences)
'''
cfp_df = pd.read_csv('./Fingerprints and Dim Reduction Dataset/cfp_cleaned_dataset.csv', index_col=0)
reducer = UMAP(n_components=2)
umap_features = reducer.fit_transform(cfp_df)
umap_features = pd.DataFrame(umap_features)
umap_features.index = cfp_df.index

umap_features.to_csv('./Fingerprints and Dim Reduction Dataset/dim_red_full_dataset.csv', index=True)

'''
Dim red FPs to individual datasets
'''
umap_features = pd.read_csv('./Fingerprints and Dim Reduction Dataset/dim_red_full_dataset.csv', index_col=0)
wang_2020 = pd.read_csv('./Caco-2 Permeability Dataset/Wang_et_al2020.csv')
wang_2016 = pd.read_csv('./Caco-2 Permeability Dataset/Wang_et_al2016.csv')
wang_chen_2020 = pd.read_csv('./Caco-2 Permeability Dataset/Wang_and_Chen2020.csv')

umap_features.reset_index(drop=False, inplace=True)
umap_features.rename(columns={'index':'SMILES'}, inplace=True)

wang_2020 = pd.merge(wang_2020, umap_features, on='SMILES', how='left')
wang_2020.dropna(axis=0, inplace=True) #NaN values derived from those SMILES entries with large differences and therefore removed
wang_2016 = pd.merge(wang_2016, umap_features, on='SMILES', how='left')
wang_2016.dropna(axis=0, inplace=True)
wang_chen_2020 = pd.merge(wang_chen_2020, umap_features, on='SMILES', how='left')
wang_chen_2020.dropna(axis=0, inplace=True)

wang_2020.to_csv('./Fingerprints and Dim Reduction Dataset/wang_2020_dim_red.csv', index=False)
wang_2016.to_csv('./Fingerprints and Dim Reduction Dataset/wang_2016_dim_red.csv', index=False)
wang_chen_2020.to_csv('./Fingerprints and Dim Reduction Dataset/wang_chen_2020_dim_red.csv', index=False)
