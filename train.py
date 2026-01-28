from    model.MDFA import Model
from    types import SimpleNamespace
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

# set cuda / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU', device)

# set parameter
labels_length = 3778    # test length
seq_length = 60         # history data length
pred_length = 1         # prediction length
batch_size = 512        # batch size 
num_epochs = 1          # number epochs

# Set model_name
model_name = ['MDFA']
save_str = model_name[0]

train_data = 0.8        # 80%
val_data = 0.1          # 10%
test_data = 0.1         # 10%

miu = 0.001             # learning rate

df = pd.read_csv('./dataset/flight_training.csv')

col = ['wind_speed', 'position_z', 'velocity_y', 'payload', 'power']
df = df[col]

print('Data head: ', df.head(), '\nData tail: ', df.tail(), '\nData shape: ', df.shape, '\nData Describe: ', df.describe())
print(df.dtypes)

# Normalization 
MinMax = MinMaxScaler((-1,1))
scaler = MinMax.fit(df)
np_alldata = scaler.transform(df)
joblib.dump(scaler, 'scaler.save') # save norm

df_data = pd.DataFrame(np_alldata, columns=df.columns)
print('data head: ',df_data.head(), 'data min: ', df_data.min(), 'data max: ', df_data.max())

# data loader
train_size = int((len(df_data)-labels_length) * (1-val_data-test_data))
val_size = int((len(df_data) - labels_length) - train_size)
print('\nData length: ', len(df_data), '\nTrain data size: ', train_size, '\nVal data size: ', val_size)

df_train = pd.DataFrame(df_data.iloc[0:train_size, :])
df_valid = pd.DataFrame(df_data.iloc[train_size:train_size+val_size, :])
df_test = pd.DataFrame(df_data.iloc[train_size+val_size:, :])

print('\nData length: ', len(df_train), '\nData head: ', df_train.head(), '\nData tail: ', df_train.tail())
print('\nData length: ', len(df_valid), '\nData head: ', df_valid.head(), '\nData tail: ', df_valid.tail())

# create dataset
np_train = df_train.to_numpy()
np_valid = df_valid.to_numpy()
np_test = df_test.to_numpy()

X_train, y_train = create_dataset(np_train,np_train[:,-1],seq_length, pred_length)
X_valid, y_valid = create_dataset(np_valid,np_valid[:,-1],seq_length, pred_length)
X_test, y_test = create_dataset(np_test,np_test[:,-1],seq_length, pred_length)
print('X_train:', X_train.shape, 'y_train:', y_train.shape, 'X_valid:', X_valid.shape, 'y_valid:', y_valid.shape)

# load model
configs = {'seq_len': seq_length,
           'label_len': 48,
           'pred_len': pred_length,
           'output_attention': False,
           'moving_avg': [1, 3, 5, 7],
           'enc_in': X_train.shape[2],
           'dec_in': X_train.shape[2], 
           'embed': 'timeF',
           'freq': 's',
           'dropout': 0.05,
           'factor': 1,
           'd_model': X_train.shape[1],
           'n_heads': 8,
           'd_ff': X_train.shape[1]*2,
           'p': 2,
           'activation': 'gelu',
           'e_layers': 2,
           'd_layers': 1,
           'c_out': 1,
           'modes':32,
           'mode_select':'random'
           }
configs = SimpleNamespace(**configs)

print('Model name:', save_str)
model = Model(configs)

model.eval()
model.to(device)
total_params = sum(p.numel() for p in model.parameters())       # parameters model
print(f"Total Parameters: {total_params:,}")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=miu)

# Training parameters
train_losses = []
valid_losses = []

print(f"Length dataloader: {len(train_loader)} {len(valid_loader)} {len(test_loader)}")

# Training loop
best_valid_loss = float('inf')
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        # print(X_batch.shape, y_batch.shape)

        optimizer.zero_grad()
        # outputs = model(X_batch)
        outputs = model(X_batch,X_batch,X_batch,X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item() * X_batch.size(0)

    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    epoch_valid_loss = 0.0
    val_predictions = []
    val_actuals = []
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch,X_batch,X_batch,X_batch)
            loss = criterion(outputs, y_batch)
            epoch_valid_loss += loss.item() * X_batch.size(0)
            val_predictions.extend(outputs.cpu().numpy().flatten())
            val_actuals.extend(y_batch.cpu().numpy().flatten())

    epoch_valid_loss /= len(valid_loader.dataset)
    valid_losses.append(epoch_valid_loss)

    # Calculate validation metrics
    val_mae = np.mean(np.abs(np.array(val_predictions) - np.array(val_actuals)))
    val_mape = np.mean(np.abs((np.array(val_actuals) - np.array(val_predictions)) / np.array(val_actuals))) * 100
    val_r2 = 1 - (np.sum((np.array(val_actuals) - np.array(val_predictions))**2) / np.sum((np.array(val_actuals) - np.mean(val_actuals))**2))
    val_rmse = np.sqrt(epoch_valid_loss)

    # Check if this is the best model so far
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        best_epoch = epoch + 1
        
        # Save the model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_valid_loss,
            'mae': val_mae,
            'mape': val_mape,
            'r2': val_r2,
            'rmse': val_rmse
        }, f'{save_str}_model_{seq_length}_{pred_length}_{X_train.shape[2]}.pth') # save model path
        print(f"Model saved at epoch {epoch+1} with validation loss: {epoch_valid_loss:.4f}")

    # if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}')
    print(f'Valid MAE:  {val_mae:.4f}, Valid MAPE: {val_mape:.2f}%, Valid R2: {val_r2:.4f}, Valid RMSE: {val_rmse:.4f}')

print(f"\nBest model saved at epoch {best_epoch} with validation loss: {best_valid_loss:.4f}")