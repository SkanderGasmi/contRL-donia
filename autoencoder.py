import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_autoencoder(self, data_loader, epochs, device):
        criterion = nn.MSELoss()
        self.to(device)  # Move model to device
        for epoch in range(epochs):
            epoch_loss = 0
            for data in data_loader:
                inputs = data[0].to(device)  # Move data to device
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(data_loader):.4f}')
