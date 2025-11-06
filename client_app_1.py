import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Reuse your Conv1dAutoencoder and PowerTraceDataset classes here
class PowerTraceDataset(Dataset):
    """
    Custom PyTorch Dataset for windowed power trace data.
    Converts time-series into overlapping windows of fixed size.
    """
    
    def __init__(self, power_values, window_size=100, overlap=0.5):
        """
        Args:
            power_values: numpy array of power consumption values (1D)
            window_size: number of timesteps per window (e.g., 100)
            overlap: fraction of overlap between consecutive windows (0.5 = 50% overlap)
        """
        self.window_size = window_size
        self.power_values = power_values
        
        # Calculate stride based on overlap
        # overlap=0.5 means stride = window_size * (1 - 0.5) = window_size/2
        self.stride = int(window_size * (1 - overlap))
        
        # Generate all valid window starting indices
        self.num_windows = (len(power_values) - window_size) // self.stride + 1
    
    def __len__(self):
        """Return total number of windows."""
        return self.num_windows
    
    def __getitem__(self, idx):
        """
        Retrieve one window at index idx.
        Returns: window as tensor of shape (1, window_size)
        """
        start = idx * self.stride
        end = start + self.window_size
        
        window = self.power_values[start:end]
        
        # Convert to tensor with shape (channels=1, sequence_length)
        window_tensor = torch.FloatTensor(window).unsqueeze(0)
        
        return window_tensor
    
class Conv1dAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1dAutoencoder, self).__init__()
        
        # Encoder compresses input time-series into smaller latent representation
        self.encoder = nn.Sequential(
            # Conv1d: input channels=1, out channels=16, kernel size=5, padding=2 to retain length
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),                     # Non-linearity after convolution
            nn.MaxPool1d(kernel_size=2),  # Downsample sequence length by factor of 2
            
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Decoder reconstructs the sequence from latent space
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # Upsample to double length
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()  # Output bounded between 0 and 1 (matching normalized input)
        )
    
    def forward(self, x):
        x_enc = self.encoder(x)  # Encode input
        x_dec = self.decoder(x_enc)  # Decode back to original shape
        return x_dec

# Example: Load benign data and simulate data splits for clients
benign_data = pd.read_csv("benign_power_traces.csv")

# Assume client_data_splits is a dict: client_id -> (train_loader)
client_data_splits = {}

num_clients = 5
window_size = 100
overlap = 0.5
batch_size = 16

for client_id in range(num_clients):
    device_df = benign_data[benign_data['device_id'] == client_id]
    power_values = device_df['power_consumption_mW'].values
    power_min = power_values.min()
    power_max = power_values.max()
    power_normalized = (power_values - power_min) / (power_max - power_min)
    dataset = PowerTraceDataset(power_normalized, window_size=window_size, overlap=overlap)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    client_data_splits[client_id] = loader


class CAEClient(NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        # Convert model parameters to numpy arrays for Flower server
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # Load parameters from Flower server to model
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update model and train on local data for 1 epoch
        self.set_parameters(parameters)
        self.model.train()
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch)
            loss.backward()
            self.optimizer.step()
        # Return updated parameters and number of training examples
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        # Optional: evaluate on local data
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.train_loader:
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(self.train_loader.dataset)
        print(f"Client evaluation loss: {avg_loss:.6f}")
        return float(avg_loss), len(self.train_loader.dataset), {"mse": float(avg_loss)}


def client_fn(context: Context):
    """
    Factory creating a FL client per device partition.
    Flower passes context which includes the partition-id.
    """
    partition_id = context.node_config["partition-id"]
    train_loader = client_data_splits[partition_id]
    model = Conv1dAutoencoder()
    client = CAEClient(model=model, train_loader=train_loader)
    print(f"Starting client {partition_id} with {len(train_loader.dataset)} training samples")
    return client


app = ClientApp(client_fn=client_fn)
