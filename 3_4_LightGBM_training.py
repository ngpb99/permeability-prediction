import pandas as pd
import optuna
import numpy as np
from utils.lgbm_utils import run_tuning, model_testing, get_rdkit_descs, get_cfps, get_bitfps

train = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/train_data.csv')
test = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/test_data.csv')
full_data = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/3D_optimizable_dataset.csv')

'''
Feature calculation
'''
# Calc rdkit desc
# rdkit_desc = get_rdkit_descs(full_data)
# rdkit_desc.dropna(inplace=True, axis=0) # 7 datapoints has NaN. 
# rdkit_desc.reset_index(drop=True, inplace=True)
# test_rdkit = pd.merge(test, rdkit_desc, on='SMILES', how='inner')
# train_rdkit = pd.merge(train, rdkit_desc, on='SMILES', how='inner')
# train_rdkit.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/rdkit_train.csv', index=False)
# test_rdkit.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/rdkit_test.csv', index=False) # For analysis whether removed data affect chemical space
train_rdkit = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/rdkit_train.csv')
test_rdkit = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/rdkit_test.csv')

# Calc count fingerprints
cfps = get_cfps(full_data)
test_cfps = pd.merge(test, cfps, on='SMILES', how='left')
train_cfps = pd.merge(train, cfps, on='SMILES', how='left')

# Calc bit fingerprints
bitfps = get_bitfps(full_data)
test_bitfps = pd.merge(test, bitfps, on='SMILES', how='left')
train_bitfps = pd.merge(train, bitfps, on='SMILES', how='left')

# Mordred descriptors
test_mordred = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_test.csv')
train_mordred = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_train.csv')


'''
LightGBM hyperparameter tuning
'''
train_list = [train_rdkit, train_cfps, train_bitfps, train_mordred]
names = ['rdkit', 'cfps', 'bitfps', 'mordred']
params_dict = {}

for i in range(len(train_list)):
    train_data = train_list[i]
    
    def objective(trial):
        params = {'num_leaves': trial.suggest_int('num_leaves', 30, 65),
                  'min_child_samples': trial.suggest_int('min_child_samples', 20, 65),
                  'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                  'learning_rate':trial.suggest_float('learning_rate', 0.01, 0.1)}
        mse = run_tuning(train_data, params)
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    trial_ = study.best_trial
    params_dict[names[i]] = trial_.params
    print('best trial:', trial_.values)
    print(f'Best parameters: {trial_.params}')
    
'''
LightGBM testing
'''

test_list = [test_rdkit, test_cfps, test_bitfps, test_mordred]
r2_results = {}
mae_results = {}
mse_results = {}

for i in range(len(test_list)):
    train_data = train_list[i]
    test_data = test_list[i]
    best_params = params_dict[names[i]]
    
    r2_list = []
    mae_list = []
    mse_list = []
    
    for n in range(5):
        r2, mse, mae = model_testing(train_data, test_data, best_params, names[i], n)
        r2_list.append(r2)
        mae_list.append(mae)
        mse_list.append(mse)
    
    r2_std, r2_mean = np.std(r2_list), np.mean(r2_list)
    mae_std, mae_mean = np.std(mae_list), np.mean(mae_list)
    mse_std, mse_mean = np.std(mse_list), np.mean(mse_list)
    
    r2_results[names[i]] = f'{r2_mean} ± {r2_std}'
    mae_results[names[i]] = f'{mae_mean} ± {mae_std}'
    mse_results[names[i]] = f'{mse_mean} ± {mse_std}'


'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------
LightGBM Tuning with only Mordred 3D descriptors
'''
train_mordred_3D = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_3D_train.csv')
test_mordred_3D = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/mordred_3D_test.csv')

def objective(trial):
    params = {'num_leaves': trial.suggest_int('num_leaves', 30, 65),
              'min_child_samples': trial.suggest_int('min_child_samples', 20, 65),
              'n_estimators': trial.suggest_int('n_estimators', 50, 500),
              'learning_rate':trial.suggest_float('learning_rate', 0.01, 0.1)}
    mse = run_tuning(train_mordred_3D, params)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
trial_ = study.best_trial
print('best trial:', trial_.values)
print(f'Best parameters: {trial_.params}')

'''
LightGBM Tuning with only Mordred 3D descriptors
'''

best_params = trial_.params
r2_list = []
mae_list = []
mse_list = []
for n in range(5):
    r2, mse, mae = model_testing(train_mordred_3D, test_mordred_3D, best_params, 'mordred_3D', n)
    r2_list.append(r2)
    mae_list.append(mae)
    mse_list.append(mse)

r2_std, r2_mean = np.std(r2_list), np.mean(r2_list)
mae_std, mae_mean = np.std(mae_list), np.mean(mae_list)
mse_std, mse_mean = np.std(mse_list), np.mean(mse_list)

print(f'r2 score: {r2_mean} ± {r2_std}')
print(f'mae score: {mae_mean} ± {mae_std}')
print(f'mse score: {mse_mean} ± {mse_std}')