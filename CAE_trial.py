import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

print(torch.__version__)
print(torch.cuda.is_available())

benign_data = pd.read_csv("benign_power_traces.csv")

print(benign_data.head())
print(f"Shape: {benign_data.shape}")
print(f"coloumns: {benign_data.columns.tolist()}")
print(f"Data types:\n {benign_data.dtypes}")

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

def plot_comparison(original, reconstructed, title='Window Reconstruction'):
    plt.figure(figsize=(8, 3))
    plt.plot(original, label='Original', marker='o')
    plt.plot(reconstructed, label='Reconstructed', marker='x')
    plt.title(title)
    plt.legend()
    plt.show()

# Step 1: Filter data for a single device
device_id_to_test = 0  # Choose the device ID you want to test on
device_df = benign_data[benign_data['device_id'] == device_id_to_test]

print(f"Data points for device {device_id_to_test}: {device_df.shape[0]}")

# Step 2: Extract power values for this device
power_values_device = device_df['power_consumption_mW'].values

# Step 3: Normalize with global stats of that device's benign power values
power_min = power_values_device.min()
power_max = power_values_device.max()
power_normalized = (power_values_device - power_min) / (power_max - power_min)

# Step 4: Create dataset and dataloader as before
dataset = PowerTraceDataset(power_normalized, window_size=100, overlap=0.5)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

print(f"Total windows for device {device_id_to_test}: {len(dataset)}")
sample_batch = next(iter(train_loader))
print(f"Sample batch shape (batch_size, channels, seq_len): {sample_batch.shape}")


model = Conv1dAutoencoder()

criterion =nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=1e-3)

num_epochs = 20  # Enough for test; change as needed

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    epoch_loss = 0

    for batch in train_loader:  # Batch shape: (batch_size, 1, 100)
        outputs = model(batch)  # Run forward pass
        loss = criterion(outputs, batch)  # Compare output to input

        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Auto-diff computes new gradients
        optimizer.step()        # Update weights
        epoch_loss += loss.item() * batch.size(0)
    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# Visualize after FIRST epoch
model.eval()
with torch.no_grad():
    first_batch = next(iter(train_loader))
    outputs = model(first_batch)
    for i in range(min(3, len(first_batch))):  # first 3 windows in this batch
        plot_comparison(first_batch[i, 0].numpy(), outputs[i, 0].numpy(), 
            title=f'First Epoch: Window {i+1}')

# Visualize after LAST epoch
model.eval()
with torch.no_grad():
    last_batch = next(iter(train_loader))  # Optionally, you could loop through to sample last batch
    outputs = model(last_batch)
    for i in range(min(3, len(last_batch))):
        plot_comparison(last_batch[i, 0].numpy(), outputs[i, 0].numpy(), 
            title=f'Last Epoch: Window {i+1}')