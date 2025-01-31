import os
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Dataset
from tqdm import tqdm
import deepchem as dc
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool


class graph_features(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super(graph_features, self).__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
    
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_chirality=True)
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(row['SMILES'])
            f = featurizer._featurize(mol)
            data = f.to_pyg_graph()
            data.y = self._get_label(row['logPapp'])
            data.smiles = row['SMILES']
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))
                
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]
    
    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
    
class engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
    
    @staticmethod
    def loss_fn(targets, outputs):
        return nn.MSELoss()(outputs, targets)
    
    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)
    
    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
                final_loss += loss.item()
        return final_loss / len(data_loader)
    
    def test(self, data_loader):
        self.model.eval()
        mae_total = 0
        mse_total = 0
        r2_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1).float(), outputs.float())
                loss = loss.item()
                
                mae = mean_absolute_error(
                    data.y.unsqueeze(1).to('cpu').detach().numpy(),
                    outputs.to('cpu').detach().numpy()
                )
                mae_total += mae

                mse = mean_squared_error(
                    data.y.unsqueeze(1).to('cpu').detach().numpy(),
                    outputs.to('cpu').detach().numpy()
                )
                mse_total += mse

                r2 = r2_score(
                    data.y.unsqueeze(1).to('cpu').detach().numpy(),
                    outputs.to('cpu').detach().numpy()
                )
                r2_total += r2

        return (
            mae_total / len(data_loader),
            mse_total / len(data_loader),
            r2_total / len(data_loader),
        )
    
    
class GAT(torch.nn.Module):
    def __init__(
        self,
        num_features,
        output,
        num_layers,
        hidden_size,
        n_heads,
        dropout,
        edge_dim,
    ):
        super(GAT, self).__init__()
        layers = []
        for layer in range(num_layers):
            if layer == 0:
                layers.append(
                    GATv2Conv(
                        in_channels=num_features,
                        out_channels=hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    )
                )
            else:
                layers.append(
                    GATv2Conv(
                        in_channels=hidden_size * n_heads,
                        out_channels=hidden_size,
                        heads=n_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    )
                )
        self.model = nn.Sequential(*layers)
        self.ro = Sequential(
            Linear(hidden_size * n_heads, hidden_size),
            ReLU(),
            Linear(hidden_size, output),
        )
    def forward(self, x, edge_attr, edge_index, batch_index):
        for layer in self.model:
            x = layer(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch_index)
        return self.ro(x)


def model_tuning(params, train_loader, valid_loader, tuning_only=True, model_path=None, repetition=None, fold_no=None):
    if not tuning_only and (model_path is None or repetition is None or fold_no is None):
        raise ValueError('model_path and repetition must be specified when tuning_only=False')
    
    num_features = 32
    output = 1
    edge_dim = 11
    model = GAT(num_features, output, params['num_layers'], 
                params['hidden_size'], params['n_heads'], 
                params['dropout'], edge_dim)
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine(model, optimizer, device='cuda')
    best_loss = np.inf
    current_iterations = 0
    early_stopping_iterations = 10
    
    if tuning_only:
        for i in range(300):
            train_loss = eng.train(train_loader)
            valid_loss = eng.validate(valid_loader)
            print(f'Epoch: {i+1}/300, train_loss: {train_loss}, valid_loss: {valid_loss}')
            if best_loss > valid_loss:
                best_loss = valid_loss
                current_iterations = 0  
            else:
                current_iterations += 1
            if current_iterations == early_stopping_iterations:
                print('Commencing early stopping...')
                break
        print(f'Early stop counter: {current_iterations}')
        return best_loss
    
    else:
        for i in range(300):
            train_loss = eng.train(train_loader)
            valid_loss = eng.validate(valid_loader)
            print(f'Epoch: {i+1}/300, train_loss: {train_loss}, valid_loss: {valid_loss}')
            if best_loss > valid_loss:
                best_loss = valid_loss
                current_iterations = 0
                torch.save(model.state_dict(), os.path.join(model_path, f'trained_GATv2_model_repeat_{repetition}_fold_{fold_no}.pt'))
            else:
                current_iterations += 1
            if current_iterations == early_stopping_iterations:
                print('Commencing early stopping...')
                break
        print(f'Early stop counter: {current_iterations}')


def model_testing(params, test_loader, model_path, repetition, fold_no):
    num_features = 32
    output = 1
    edge_dim = 11
    model_path = os.path.join(model_path, f'trained_GATv2_model_repeat_{repetition}_fold_{fold_no}.pt')
    model = GAT(num_features, output, params['num_layers'], 
                params['hidden_size'], params['n_heads'], 
                params['dropout'], edge_dim)
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(),lr = params['learning_rate'])
    eng = engine(model = model, optimizer = optimizer, device = 'cuda')
    mae, mse, r2 = eng.test(test_loader)
    return mae, mse, r2