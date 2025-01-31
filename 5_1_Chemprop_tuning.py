import subprocess

hyp_opt = [
    'chemprop_hyperopt',
    '--data_path', './Caco-2 Permeability Dataset/Filtered Data/Data Split/train_data.csv',
    '--dataset_type', 'regression', 
    '--num_iters', '50',
    '--config_save_path', './Trained Models/Chemprop/chemprop_combined_dataset_optimized_hyperparameters.json',
    '--gpu', '0',
    '--metric', 'mse',
    '--extra_metrics', 'mae', 'r2'
    ]

result = subprocess.run(hyp_opt, capture_output=True, text=True)
if result.returncode == 0:
    print("Training completed successfully.")
else:
    print("Error occurred during training.")
    print(f"Return code: {result.returncode}")
