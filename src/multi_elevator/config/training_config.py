"""
Configuration settings for training the Multi-Elevator RL model.

This file defines the core parameters for training, evaluation, and file paths.
These settings can be overridden by command-line arguments in the training script.
"""
from pathlib import Path
from typing import Dict, Any

# --- Base Directory Paths ---
# Define the root directory of the project to anchor all other paths.
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
TENSORBOARD_DIR = BASE_DIR / "logs" / "tensorboard"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs" / "training"

# --- Training Configuration ---
TRAINING_CONFIG: Dict[str, Any] = {
    # --- Environment Parameters ---
    "num_envs": 8,                   # Number of parallel environments to use for training
    "num_elevators": 3,              # Number of elevators in the environment
    "num_floors": 5,                 # Number of floors in the environment
    "max_steps": 500,                # Maximum steps per episode before truncation
    "passenger_rate": 0.1,           # Probability of a new passenger arriving at any floor per step

    # --- Training Control Parameters ---
    "total_timesteps": 1_000_000,    # Total number of timesteps to train the model
    "eval_freq": 10000,              # Run evaluation every N timesteps
    "n_eval_episodes": 10,           # Number of episodes to run for each evaluation

    # --- PPO Hyperparameters ---
    # For detailed explanations, see the Stable Baselines3 PPO documentation.
    "learning_rate": 3e-4,           # Step size for the optimizer
    "n_steps": 2048,                 # Number of steps to run for each environment per update
    "batch_size": 64,                # Number of samples in each mini-batch
    "n_epochs": 10,                  # Number of optimization epochs per update
    "gamma": 0.99,                   # Discount factor for future rewards
    "gae_lambda": 0.95,              # Factor for trade-off of bias vs. variance for GAE
    "clip_range": 0.2,               # Clipping parameter for the PPO policy loss
    "clip_range_vf": None,           # Clipping parameter for the value function loss (None = disabled)
    "ent_coef": 0.01,                # Entropy coefficient for the loss calculation (encourages exploration)
    "vf_coef": 0.5,                  # Value function coefficient for the loss calculation
    "max_grad_norm": 0.5,            # Maximum value for the gradient clipping
    "target_kl": 0.01,               # Limit the KL divergence between updates
    "stats_window_size": 100,        # Window size for the rollout logging,

    # --- Policy Network Architecture ---
    # Defines the structure of the policy (pi) and value (vf) networks.
    "policy_kwargs": {
        "net_arch": {
            "pi": [64, 64],          # Two hidden layers with 64 units each for the policy network
            "vf": [64, 64]           # Two hidden layers with 64 units each for the value network
        }
    }
}

# --- Evaluation Configuration ---
EVAL_CONFIG: Dict[str, Any] = {
    "num_episodes": 100,             # Number of episodes to run during a full evaluation
    "render": False,                 # Whether to render the environment during evaluation
    "deterministic": True            # Whether to use deterministic actions during evaluation
}

# --- Model and Checkpoint Paths ---
MODEL_PATHS = {
    "best_model": MODEL_DIR / "best_models" / "best_model.zip",
    "latest_model": MODEL_DIR / "best_models" / "latest_model.zip",
    "checkpoints": MODEL_DIR / "checkpoints"
}

# --- Directory Creation ---
# Ensure that all necessary directories exist before starting the training process.
for path in [TENSORBOARD_DIR, LOGS_DIR, MODEL_PATHS["checkpoints"], MODEL_DIR / "best_models"]:
    path.mkdir(parents=True, exist_ok=True) 