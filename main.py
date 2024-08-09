import time
import torch
from autoencoder import Autoencoder
from novelty_detector import NoveltyDetector
from training_pipeline import TrainingPipeline
from utils import setup_environment, process_observations, handle_anomaly_detection, update_models_if_needed, play_environment

# Initialize GPU/TPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize sets for AE and PPO models, thresholds, and environment names
ae_set = []
ppo_set = []
thresholds = []
environments = ["MiniGrid-LavaGapS7-v0", "MiniGrid-DoorKey-6x6-v0", "MiniGrid-Dynamic-Obstacles-5x5-v0"]

# Parameters
num_episodes = 1
num_steps_exploring = 1
num_steps_playing = 1

def main():
    prev_name = "None"
    start_time = time.time()

    for episode in range(num_episodes):
        env_name = environments[episode % len(environments)]
        env_handler = setup_environment(env_name)
        log_progress(f"Episode number: {episode + 1}. Current Environment in this episode: {env_name}")

        # Collect observations
        observations_tensor, scaler = process_observations(env_handler, num_steps_exploring)
        collected_observations_ae = torch.stack([move_to_device(obs, device) for obs in observations_tensor])

        # Anomaly detection
        if episode == 0:
            change_in_environment = True
        else:
            change_in_environment = handle_anomaly_detection(collected_observations_ae, episode, prev_name, env_name, ae_model, threshold)

        if change_in_environment:
            isNew, best_autoencoder, lowest_error, best_index = Novelty_recognition(
                collected_observations_ae, ae_set, thresholds
            )
            ppo_model, ae_model, threshold = update_models_if_needed(
                isNew, collected_observations_ae, env_name, env_handler, ae_set, ppo_set, thresholds, device
            )

        # Play the environment
        total_reward = play_environment(env_handler, ppo_model, num_steps_playing, device)
        log_progress(f"Episode {episode + 1}: Total Reward = {total_reward}")

        prev_name = env_name

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_progress(f"Total elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
