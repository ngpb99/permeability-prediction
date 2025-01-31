from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from rdkit import Chem
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator, Descriptors


def run_tuning(train_data, params):
    x = train_data.drop(columns=['SMILES', 'logPapp'])
    y = train_data['logPapp']
    
    kfold = KFold(n_splits=5)
    lgb_model = LGBMRegressor(num_leaves=params['num_leaves'], 
                              min_child_samples=params['min_child_samples'], 
                              n_estimators=params['n_estimators'],
                              learning_rate=params['learning_rate'])
    total_mse = 0
    for fold_no, (train_idx, valid_idx) in enumerate(kfold.split(x)):
        x_train, x_valid = x.iloc[train_idx,:], x.iloc[valid_idx,:]
        y_train, y_valid = y[train_idx], y[valid_idx]
        lgb_model.fit(x_train, y_train)
        y_pred = lgb_model.predict(x_valid)
        mse = mean_squared_error(y_valid, y_pred)
        total_mse += mse
    return mse/5

def model_testing(train_data, test_data, best_params, names, n_reps):
    x_train = train_data.drop(columns=['SMILES', 'logPapp'])
    y_train = train_data['logPapp']
    x_test = test_data.drop(columns=['SMILES', 'logPapp'])
    y_test = test_data['logPapp']
    
    lgb_model = LGBMRegressor(num_leaves=best_params['num_leaves'], 
                              min_child_samples=best_params['min_child_samples'], 
                              n_estimators=best_params['n_estimators'],
                              learning_rate=best_params['learning_rate'])
    lgb_model.fit(x_train, y_train)
    
    if names == 'rdkit':
        joblib.dump(lgb_model, f'./Trained Models/LightGBM/RDKit Descriptors/{names}_trained_model_repeat_{n_reps}.pkl')
    elif names == 'cfps':
        joblib.dump(lgb_model, f'./Trained Models/LightGBM/Count Fingerprints/{names}_trained_model_repeat_{n_reps}.pkl')
    elif names == 'bitfps':
        joblib.dump(lgb_model, f'./Trained Models/LightGBM/Bit Fingerprints/{names}_trained_model_repeat_{n_reps}.pkl')
    elif names == 'mordred':
        joblib.dump(lgb_model, f'./Trained Models/LightGBM/Mordred Descriptors/{names}_trained_model_repeat_{n_reps}.pkl')
    elif names == 'mordred_3D':
        joblib.dump(lgb_model, f'./Trained Models/LightGBM/Mordred Descriptors/{names}_trained_model_repeat_{n_reps}.pkl')
    
    y_pred = lgb_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return r2, mse, mae

def get_rdkit_descs(data):
    smi_df = data['SMILES']
    all_descs = [x[0] for x in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descs)
    desc_list = []
    
    for smile in smi_df:
        mol = Chem.MolFromSmiles(smile)
        desc = calc.CalcDescriptors(mol)
        desc_list.append(desc)
    df_desc = pd.DataFrame(desc_list, columns=all_descs)
    
    var_per_col = list(df_desc.var())
    zero_var_desc = []
    for idx, var in enumerate(var_per_col):
        if var == 0:
            zero_var_desc.append(idx)
    df_desc = df_desc.drop(columns=df_desc.columns[zero_var_desc])
    df_desc['SMILES'] = data['SMILES']
    return df_desc
    
def get_cfps(data):
    smi_df = data['SMILES']
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    cfp_list = []
    
    for smile in smi_df:
        mol = Chem.MolFromSmiles(smile)
        cfp = mfpgen.GetCountFingerprint(mol)
        cfp = list(cfp)
        cfp_list.append(cfp)
    df_cfp = pd.DataFrame(cfp_list)
    df_cfp['SMILES'] = data['SMILES']
    return df_cfp

def get_bitfps(data):
    smi_df = data['SMILES']
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    bitfp_list = []
    
    for smile in smi_df:
        mol = Chem.MolFromSmiles(smile)
        fps =  mfpgen.GetFingerprint(mol)
        fps = list(fps)
        bitfp_list.append(fps)
    df_bitfp = pd.DataFrame(bitfp_list)
    df_bitfp['SMILES'] = data['SMILES']
    return df_bitfp
