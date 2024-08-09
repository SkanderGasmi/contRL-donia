import torch
from environment_handler import EnvironmentHandler
from utils import move_to_device, normalize_observations, log_progress
from anomaly_detection import recognize_anomaly, Novelty_recognition, train_PPO_AE

def setup_environment(env_name):
    """Initialize the environment handler."""
    return EnvironmentHandler(env_name)

def process_observations(env_handler, num_steps):
    """Collect and normalize observations."""
    observations = env_handler.collect_observations(num_steps)
    normalized_obs, scaler = normalize_observations(observations)
    return torch.tensor(normalized_obs, dtype=torch.float32), scaler

def handle_anomaly_detection(observations_tensor, episode, prev_name, current_env_name, ae_model, threshold):
    """Detect if there is an anomaly in the environment."""
    if episode == 0:
        return True

    filename = f"AnomalyCheck_prev_{prev_name}_curr_{current_env_name}"
    collected_observations_cpu = observations_tensor.cpu().numpy()
    return recognize_anomaly(
        collected_observations_cpu, ae_model, threshold, filename=filename
    )

def update_models_if_needed(isNew, collected_observations_ae, current_env_name, env_handler, ae_set, ppo_set, thresholds, device):
    """Update AE and PPO models if a new environment is detected."""
    if isNew:
        log_progress("Novel environment detected, retraining autoencoder and PPO.")
        ppo_model, ae_model, threshold = train_PPO_AE(env_handler.env, current_env_name, collected_observations_ae.cpu())
        ppo_model.to(device)
        ae_model.to(device)

        ae_set.append(ae_model)
        ppo_set.append(ppo_model)
        thresholds.append(threshold)
        return ppo_model, ae_model, threshold
    else:
        best_index = thresholds.index(min(thresholds))
        ppo_model = ppo_set[best_index].to(device)
        ae_model = ae_set[best_index].to(device)
        threshold = thresholds[best_index]
        return ppo_model, ae_model, threshold

def play_environment(env_handler, ppo_model, num_steps_playing, device):
    """Play the environment using the PPO model and return the total reward."""
    done = False
    play_step = 0
    total_reward = 0
    current_state, _ = env_handler.reset()
    current_state = move_to_device(torch.tensor(normalize_images(current_state), dtype=torch.float32), device)

    while not done and play_step < num_steps_playing:
        play_step += 1
        action, _ = ppo_model.predict(current_state, deterministic=True)

        next_state, reward, done, _, _ = env_handler.step(action)
        current_state = move_to_device(torch.tensor(normalize_images(next_state), dtype=torch.float32), device)
        total_reward += reward

    return total_reward
