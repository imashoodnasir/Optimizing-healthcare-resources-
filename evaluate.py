import torch
from torch.utils.data import DataLoader
from model import LSTMGRUModel
from preprocessing import load_and_preprocess
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset, scaler = load_and_preprocess('data/hospital_dataset.csv')

train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
_, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = LSTMGRUModel(input_size=dataset[0][0].shape[1], hidden_size=128, output_size=dataset[0][1].shape[0])
model.load_state_dict(torch.load('lstm_gru_model.pth'))
model.eval()

predictions, actuals = [], []

with torch.no_grad():
    for seq, target in test_loader:
        pred = model(seq)
        predictions.extend(pred.numpy())
        actuals.extend(target.numpy())

predictions = scaler.inverse_transform(predictions)
actuals = scaler.inverse_transform(actuals)

mse = mean_squared_error(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}')
