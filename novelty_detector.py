import torch
import torch.nn as nn

class NoveltyDetector:
    def __init__(self, autoencoder, threshold, device='cpu'):
        self.autoencoder = autoencoder.to(device)
        self.threshold = threshold
        self.device = device
    
    def compute_reconstruction_error(self, observations):
        self.autoencoder.eval()  # Set the model to evaluation mode
        observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        reconstructed = self.autoencoder(observations)
        errors = nn.functional.mse_loss(reconstructed, observations, reduction='none')
        errors = errors.mean(dim=(1, 2, 3))  # Adjust dimensions based on your observation shape
        return errors.cpu().numpy()  # Move back to CPU for further processing

    def detect_novelty(self, errors):
        return any(error > self.threshold for error in errors)
