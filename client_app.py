"""
CLIENT APP — for Federated CAE training
Each client represents one IoT device’s power trace dataset
"""


import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np




# -------------------------
# Dataset + Model Definition
# -------------------------
class PowerTraceDataset(Dataset):
    """Creates overlapping sliding windows from power trace sequence"""


    def __init__(self, power_values, window_size=100, overlap=0.5):
        self.window_size = window_size
        self.power_values = power_values
        self.stride = int(window_size * (1 - overlap))
        self.num_windows = (len(power_values) - window_size) // self.stride + 1


    def __len__(self):
        return self.num_windows


    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        window = self.power_values[start:end]
        return torch.FloatTensor(window).unsqueeze(0)  # (1, window_size)




class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1dAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(16, 1, 5, stride=1, padding=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        return self.decoder(self.encoder(x))




# -------------------------
# Data Loading Utilities
# -------------------------
def load_local_data(device_id, csv_path="benign_power_traces.csv"):
    """Loads local data for a specific IoT device ID"""
    df = pd.read_csv(csv_path)
    device_df = df[df["device_id"] == device_id]


    power_values = device_df["power_consumption_mW"].values
    power_min, power_max = power_values.min(), power_values.max()
    normalized = (power_values - power_min) / (power_max - power_min)


    dataset = PowerTraceDataset(normalized, window_size=100, overlap=0.5)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader




# -------------------------
# Training / Evaluation
# -------------------------
def train(model, loader, epochs=3, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"[Local Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    return avg_loss




def evaluate(model, loader):
    criterion = nn.MSELoss()
    model.eval()
    loss_total = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            loss_total += criterion(outputs, batch).item() * batch.size(0)
    return loss_total / len(loader.dataset)




# -------------------------
# Flower Client Definition
# -------------------------
class CAEClient(NumPyClient):
    """Federated client wrapping the local CAE model"""


    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader


    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]


    def set_parameters(self, parameters):
        state_dict = dict(
            zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters])
        )
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train(self.model, self.train_loader, epochs=3)
        return self.get_parameters(), len(self.train_loader.dataset), {"train_loss": loss}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = evaluate(self.model, self.test_loader)
        print(f"[Eval] Local validation loss: {loss:.6f}")
        return float(loss), len(self.test_loader.dataset), {"val_loss": float(loss)}




# -------------------------
# Client Factory Function
# -------------------------
def client_fn(context: Context):
    """
    Factory function Flower calls to create each client instance.
    Each client trains on its assigned IoT device data partition.
    """
    partition_id = context.node_config["partition-id"]
    print(f"🚀 Starting client for device_id {partition_id}")


    train_loader = load_local_data(device_id=partition_id)
    test_loader = load_local_data(device_id=partition_id)


    model = Conv1dAutoencoder()
    return CAEClient(model, train_loader, test_loader).to_client()




# -------------------------
# Create the Client App
# -------------------------
app = ClientApp(client_fn=client_fn)



