import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess(filepath):
    data = pd.read_csv(filepath)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)

    sequence_length = 10
    sequences, targets = [], []

    for i in range(len(data_scaled) - sequence_length):
        sequences.append(data_scaled[i:i+sequence_length])
        targets.append(data_scaled[i+sequence_length])

    sequences = torch.tensor(sequences, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    dataset = TensorDataset(sequences, targets)
    return dataset, scaler
