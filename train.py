import torch
from torch.utils.data import DataLoader
from model import LSTMGRUModel
from preprocessing import load_and_preprocess
import torch.optim as optim
import torch.nn as nn

dataset, scaler = load_and_preprocess('data/hospital_dataset.csv')

train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = LSTMGRUModel(input_size=dataset[0][0].shape[1], hidden_size=128, output_size=dataset[0][1].shape[0])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00055)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for seq, target in train_loader:
        optimizer.zero_grad()
        pred = model(seq)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'lstm_gru_model.pth')
