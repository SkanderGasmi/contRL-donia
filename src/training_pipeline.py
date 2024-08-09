import numpy as np
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch

class TrainingPipeline:
    def __init__(self, env_name, autoencoder, novelty_detector):
        self.env_handler = EnvironmentHandler(env_name)
        self.autoencoder = autoencoder
        self.novelty_detector = novelty_detector
    
    def run(self, num_steps, autoencoder_epochs, ppo_epochs):
        # Collect observations
        print("Collecting observations from the environment...")
        observations = self.env_handler.collect_observations(num_steps)
        
        # Normalize observations
        scaler = StandardScaler()
        normalized_obs = scaler.fit_transform(observations)
        data_loader = DataLoader(TensorDataset(torch.tensor(normalized_obs, dtype=torch.float32)), batch_size=64, shuffle=True)
        
        # Train autoencoder
        print("Training autoencoder...")
        self.autoencoder.train_autoencoder(data_loader, autoencoder_epochs)
        
        # Detect novelty
        print("Detecting novelty...")
        errors = self.novelty_detector.compute_reconstruction_error(normalized_obs)
        is_novel = self.novelty_detector.detect_novelty(errors)
        
        if is_novel:
            print("Novel environment detected, retraining autoencoder.")
            # Update thresholds and retrain if necessary
        
        # Train PPO agent
        print("Training PPO agent...")
        ppo_model = PPO('MlpPolicy', self.env_handler.env, verbose=1)
        ppo_model.learn(total_timesteps=ppo_epochs)
        print("PPO training complete.")
