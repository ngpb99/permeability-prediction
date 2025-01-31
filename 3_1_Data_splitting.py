from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('./Caco-2 Permeability Dataset/Filtered Data/3D_optimizable_dataset.csv')
X = data['SMILES']
y = data['logPapp']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
train.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/train_data.csv', index=False)
test.to_csv('./Caco-2 Permeability Dataset/Filtered Data/Data Split/test_data.csv', index=False)