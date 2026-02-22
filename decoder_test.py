import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.trace_container.trace_obj import Trace

# General hyper param
lr=1e-3
latent_dim=16
window_size=50
epochs = 50

# Hyper param encoder

## CNN
in_channels_first_layer = 1
in_channels_second_layer = 16

out_channels_first_layer = 16
out_channels_second_layer = 32

kernel_size_first_layer = 5
kernel_size_second_layer = 5

padding_first_layer = 2
padding_second_layer = 2

## LSTM
input_size_lstm = 32
hidden_size_lstm = 64

## Linear layer
in_feat_linea_layer = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, signal, window_size, stride):
        self.signal = torch.tensor(signal, dtype=torch.float32)
        self.window_size = window_size # the dimension of the window
        self.stride = stride # how much the windows moves at every step

    def __len__(self):
        # In how many parts the signal is divided
        return (len(self.signal) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        window = self.signal[start:start + self.window_size]
        return window.unsqueeze(0) 


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels_first_layer, out_channels_first_layer, kernel_size=kernel_size_first_layer, padding=padding_first_layer),
            nn.ReLU(),
            nn.Conv1d(in_channels_second_layer, out_channels_second_layer, kernel_size=kernel_size_second_layer, padding=padding_second_layer),
            nn.ReLU()
        ).to(device)
        self.lstm = nn.LSTM(
            input_size=input_size_lstm,
            hidden_size=hidden_size_lstm,
            batch_first=True
        ).to(device)
        self.fc = nn.Linear(in_feat_linea_layer).to(device)

    def forward(self, x):
        # x: (B, 1, W)
        x = self.cnn(x)            # (B, 32, W)
        x = x.permute(0, 2, 1)     # (B, W, 32)
        _, (h, _) = self.lstm(x)   # h: (1, B, 64)
        z = self.fc(h[-1])         # (B, latent_dim)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Linear(latent_dim, 64).to(device)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        ).to(device)
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2)
        ).to(device)

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z)                       # (B, 64)
        x = x.unsqueeze(1).repeat(1, self.window_size, 1)
        x, _ = self.lstm(x)                  # (B, W, 32)
        x = x.permute(0, 2, 1)               # (B, 32, W)
        x = self.cnn(x)                      # (B, 1, W)
        return x


class CNNLSTMAutoencoder(nn.Module):
    def __init__(self, latent_dim, window_size):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, window_size)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

latentVec = []

model = CNNLSTMAutoencoder(latent_dim, window_size)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = nn.MSELoss()

WT_trace = Trace("./tests/WT/WT_Trace_Date_repetition.csv")
signal = torch.Tensor(WT_trace.get_column("rain")).to(device)

dataset = SlidingWindowDataset(signal, window_size=window_size, stride=window_size)
loader = DataLoader(dataset, batch_size=window_size, shuffle=False)

model.train()
for epoch in range(epochs):
    total_loss = 0
    
    for x in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        x_hat, _ = model(x)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch:03d} | Loss {total_loss / len(loader):.6f}")

model.eval()  

with torch.no_grad():  
    for x in DataLoader(dataset, batch_size=window_size, shuffle=False):
        _, z = model(x)   
        
        latentVec.append(z.cpu())

print(latentVec)







