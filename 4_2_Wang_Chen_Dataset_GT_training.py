from utils.gt_utils import graph_features, model_tuning, model_testing
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import optuna
import torch
import numpy as np

'''
hyperparams tuning
'''
def objective(trial):
    params={'num_layers': trial.suggest_int('num_layers', 1, 3),
            'hidden_size': trial.suggest_int('hidden_size', 64, 512),
            'n_heads': trial.suggest_int('n_heads', 1, 10),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 9e-3, log=True)}
    
    data = graph_features(root='./2D graph data/wang_chen_2020/train', filename='wc2020_train_data.csv')
    kf = KFold(n_splits=5)
    total_loss = 0
    
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(data)):
        print(f'Fold No.: {fold_no}')
        train_list = []
        valid_list = []
        for t in train_idx:
            train_list.append(torch.load(f'./2D graph data/wang_chen_2020/train/processed/data_{t}.pt', weights_only=False))
        for v in valid_idx:
            valid_list.append(torch.load(f'./2D graph data/wang_chen_2020/train/processed/data_{v}.pt', weights_only=False))
        
        train_loader = DataLoader(train_list, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_list, batch_size=256, shuffle=False)
        loss = model_tuning(params, train_loader, valid_loader)
        total_loss += loss
    
    return total_loss/5

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 50)
trial_ = study.best_trial
print(f'best trial:{trial_.values}')
print(f'Best parameters: {trial_.params}')


'''
model training and evaluation w/ optimal hyperparams
'''
best_params = trial_.params
model_path = './Trained Models/GATv2/wang_and_chen2020'
kf = KFold(n_splits=5)
data_train = graph_features(root='./2D graph data/wang_chen_2020/train', filename='wc2020_train_data.csv')
data_test = graph_features(root='./2D graph data/wang_chen_2020/test', filename='wc2020_test_data.csv')
test_loader = DataLoader(data_test, batch_size=256, shuffle=False)
performance = {}
for repetition in range(5):
    r2_list = []
    mae_list = []
    mse_list = []
    
    for fold_no, (train_idx, valid_idx) in enumerate(kf.split(data_train)):
        print(f'Repetition: {repetition}, Fold No.: {fold_no}')
        train_list = []
        valid_list = []
        for t in train_idx:
            train_list.append(torch.load(f'./2D graph data/wang_chen_2020/train/processed/data_{t}.pt', weights_only=False))
        for v in valid_idx:
            valid_list.append(torch.load(f'./2D graph data/wang_chen_2020/train/processed/data_{v}.pt', weights_only=False))
        
        train_loader = DataLoader(train_list, batch_size=256, shuffle=True)
        valid_loader = DataLoader(valid_list, batch_size=256, shuffle=False)
        model_tuning(params = best_params, train_loader = train_loader, valid_loader = valid_loader,
                     tuning_only = False, model_path = model_path, repetition = repetition, fold_no = fold_no)
        mae, mse, r2 = model_testing(params = best_params, test_loader = test_loader, model_path = model_path,
                                    repetition = repetition, fold_no = fold_no)
        print(f'Testing Completed for repetition: {repetition}, fold No.: {fold_no}\n')
        print('--------------------------------------------------------------------\n')
        print(f'Performance:\nMAE:{mae}\nMSE:{mse}\nR2:{r2}\n')
        r2_list.append(r2)
        mae_list.append(mae)
        mse_list.append(mse)
    
    r2_std, r2_mean = np.std(r2_list), np.mean(r2_list)
    mae_std, mae_mean = np.std(mae_list), np.mean(mae_list)
    mse_std, mse_mean = np.std(mse_list), np.mean(mse_list)
    performance[f'Repetition_{repetition}'] = f'R2: {r2_mean} ± {r2_std}, MAE: {mae_mean} ± {mae_std}, MSE: {mse_mean} ± {mse_std}'

print(performance)