"""
Configuration settings for training the Multi-Elevator RL model.
"""
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
TENSORBOARD_DIR = BASE_DIR / "ppo_multi_elevator_tensorboard"
MODEL_DIR = BASE_DIR / "best_model"
LOGS_DIR = BASE_DIR / "logs"

# Training parameters
TRAINING_CONFIG: Dict[str, Any] = {
    # Environment parameters
    "num_envs": 8,
    "num_elevators": 3,
    "num_floors": 5,
    "max_steps": 500,
    "passenger_rate": 0.1,
    
    # Training parameters
    "total_timesteps": 1_000_000,
    "eval_freq": 10000,
    "n_eval_episodes": 10,
    
    # PPO hyperparameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.01,
    "stats_window_size": 100,
    
    # Policy network architecture
    "policy_kwargs": {
        "net_arch": dict(
            pi=[64, 64],
            vf=[64, 64]
        )
    }
}

# Evaluation parameters
EVAL_CONFIG: Dict[str, Any] = {
    "num_episodes": 100,
    "render": False,
    "deterministic": True
}

# Model paths
MODEL_PATHS = {
    "best_model": MODEL_DIR / "best_model.zip",
    "latest_model": MODEL_DIR / "latest_model.zip",
    "checkpoints": MODEL_DIR / "checkpoints"
}

# Create directories if they don't exist
for path in [TENSORBOARD_DIR, MODEL_DIR, LOGS_DIR, MODEL_PATHS["checkpoints"]]:
    path.mkdir(parents=True, exist_ok=True) 