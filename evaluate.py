import  os

import  torch
import  torch.nn as nn
from    torch.utils.data import DataLoader, TensorDataset

import  pandas as pd
from    tqdm import tqdm
import  numpy as np
import  matplotlib.pyplot as plt

from    sklearn.preprocessing import MinMaxScaler, StandardScaler
import  joblib

from    utils import create_dataset, inspectData, plotAny
import  seaborn as sns

from    model.MDFA import Model
from    types import SimpleNamespace

df = pd.read_csv('./dataset/flight_197.csv')

# drop date, time_day, datetime
df['date_2'] = df['date'].astype(str)
df['time_day_2'] = df['time_day'].astype(str)
df['time_2'] = pd.to_timedelta(df['time'], unit='s')
df['timestamp'] = pd.to_datetime(df['date_2'] + ' ' + df['time_day_2']) + df['time_2']

# features
col = ['timestamp', 'wind_speed', 'position_z', 'velocity_y', 'payload', 'power']
df = df[col]

df = df.copy(deep=True)

print('Data head: ', df.head(), '\nData tail: ', df.tail(), '\nData shape: ', df.shape, '\nData Describe: ', df.describe())
print(df.std(), df.mean())

df_data = df.drop(columns=["timestamp"])

# normalize
MinMax = MinMaxScaler((-1,1))
scaler = MinMax.fit(df_data)
np_alldata = scaler.transform(df_data)

# put normalize into dump
joblib.dump(scaler, 'scaler_test.save')

df_data = pd.DataFrame(np_alldata, columns=df_data.columns)
print(df_data.head(), df_data.min(), df_data.max())

# put dataset into numpy
np_test = df_data.to_numpy()

# set parameters
labels_length = 60
pred_length = 1
X_test, y_test = create_dataset(np_test, np_test[:,-1], labels_length, pred_length)
print(X_test.shape, y_test.shape)

# set name model
model_name = ['MDFA']

# Instantiate the model -> lstm, gru, tcn
input_size = X_test.shape[2]
hidden_size1 = X_test.shape[1]
hidden_size2 = hidden_size1 * 2
hidden_size3 = hidden_size1 * 5
output_size = pred_length
miu = 0.001

# for my model
configs = {'seq_len': labels_length,
           'label_len': 48,
           'pred_len': pred_length,
           'output_attention': False,
           'moving_avg': [1, 3, 5, 7],
           'enc_in': X_test.shape[2],
           'dec_in': X_test.shape[2], 
           'embed': 'timeF',
           'freq': 's',
           'dropout': 0.05,
           'factor': 1,
           'd_model': X_test.shape[1],
           'n_heads': 8,
           'd_ff': X_test.shape[1] * 2,
           'p': 2,
           'activation': 'gelu',
           'e_layers': 2,
           'd_layers': 1,
           'c_out': 1,
           'modes':32,
           'mode_select':'random'           
           }
configs = SimpleNamespace(**configs)

model = Model(configs)

# Define loss and optimizer
criterion = nn.MSELoss()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# load model
model.to(device)

# convert data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# create datasets and dataloaders
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# training parameters
train_losses = []
valid_losses = []

print(f"Length dataloader: {len(test_loader)}")

# load model / proposed model
checkpoint = torch.load(f'{model_name[0]}_model_{labels_length}_{pred_length}_{X_test.shape[2]}.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded be {model_name[0]} model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['loss']:.4f}")

# testing
def test(model, save_str):
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if save_str in ['lstm', 'gru', 'tcn']:
                outputs = model(X_batch)
            else:
                outputs = model(X_batch, X_batch, X_batch, X_batch)

            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            if pred_length > 1:
                predictions.extend(outputs[1].cpu().numpy().flatten())
                actuals.extend(y_batch[1].cpu().numpy().flatten())
            else:
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(y_batch.cpu().numpy().flatten())

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss (MSE): {test_loss:.4f}')

    # Calculate test metrics
    test_mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    test_mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
    test_r2 = 1 - (np.sum((np.array(actuals) - np.array(predictions))**2) / np.sum((np.array(actuals) - np.mean(actuals))**2))
    test_rmse = np.sqrt(test_loss)

    print(f'Test MSE: {test_loss:.4f}')
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test MAPE: {test_mape:.2f}%')
    print(f'Test R2: {test_r2:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')

    # # Plot predictions vs actuals
    # plt.figure(figsize=(10, 5))
    # plt.plot(actuals, label='Actual')
    # plt.plot(predictions, label='Predicted')
    # plt.xlabel('Sample')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.savefig(f'{save_str}.png')
    # plt.show()

multi = test(model, model_name[0])
print(multi)