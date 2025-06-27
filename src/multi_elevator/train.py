"""
This script handles the training process for the Multi-Elevator RL agent.

It uses the Proximal Policy Optimization (PPO) algorithm from the Stable Baselines3
library to train a model on the custom `MultiElevatorEnv` environment.

The script sets up the training environment, including vectorization for parallel
processing, and configures callbacks for evaluation and checkpointing. It allows
for dynamic configuration of hyperparameters, which can be passed as arguments.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from multi_elevator.environment.env import MultiElevatorEnv
from multi_elevator.config.training_config import (
    TRAINING_CONFIG, EVAL_CONFIG, MODEL_PATHS, TENSORBOARD_DIR, LOGS_DIR
)

# Configure logging for the training process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def make_env(rank: int, seed: int = 0):
    """
    Utility function for creating a single, monitored environment instance.

    This function is used by the vectorized environment to create each parallel
    environment. It wraps the base environment with a `Monitor` to log statistics.

    Args:
        rank: A unique identifier for the environment instance.
        seed: The random seed for the environment.

    Returns:
        A callable that returns the created environment.
    """
    def _init():
        env = MultiElevatorEnv(
            num_elevators=TRAINING_CONFIG["num_elevators"],
            num_floors=TRAINING_CONFIG["num_floors"],
            max_steps=TRAINING_CONFIG["max_steps"],
            passenger_rate=TRAINING_CONFIG["passenger_rate"]
        )
        # Wrap the environment with a Monitor to log rewards and other info
        env = Monitor(env, str(LOGS_DIR / f"env_{rank}.log"))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init

def create_vec_env(num_envs: int, seed: int = 0):
    """
    Creates a vectorized environment for parallel training.

    If `num_envs` is greater than 1, it uses `SubprocVecEnv` for multiprocessing.
    Otherwise, it uses a `DummyVecEnv` for single-process training.

    Args:
        num_envs: The number of parallel environments to create.
        seed: The base random seed.

    Returns:
        The created vectorized environment.
    """
    if num_envs > 1:
        return SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    return DummyVecEnv([make_env(0, seed)])

def train(
    seed: int = 0,
    num_envs: Optional[int] = None,
    total_timesteps: Optional[int] = None,
    eval_freq: Optional[int] = None,
    n_eval_episodes: Optional[int] = None,
    learning_rate: Optional[float] = None,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_epochs: Optional[int] = None,
    gamma: Optional[float] = None,
    gae_lambda: Optional[float] = None,
    clip_range: Optional[float] = None,
    ent_coef: Optional[float] = None,
    vf_coef: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    target_kl: Optional[float] = None,
    stats_window_size: Optional[int] = None,
    policy_kwargs: Optional[dict] = None,
    session_id: Optional[str] = None
) -> None:
    """
    Main function to train the PPO model.

    This function orchestrates the training process by:
    - Setting up session-specific directories for models, logs, and checkpoints.
    - Creating training and evaluation environments.
    - Configuring callbacks for periodic evaluation and model saving.
    - Initializing and training the PPO agent.

    Args:
        seed: Random seed for reproducibility.
        num_envs: Number of parallel environments (overrides config).
        total_timesteps: Total training timesteps (overrides config).
        eval_freq: Evaluation frequency (overrides config).
        n_eval_episodes: Number of evaluation episodes (overrides config).
        learning_rate: The learning rate for the optimizer (overrides config).
        n_steps: Number of steps per environment per update (overrides config).
        batch_size: Minibatch size (overrides config).
        n_epochs: Number of optimization epochs per update (overrides config).
        gamma: Discount factor (overrides config).
        gae_lambda: GAE lambda parameter (overrides config).
        clip_range: PPO clipping parameter (overrides config).
        ent_coef: Entropy coefficient (overrides config).
        vf_coef: Value function coefficient (overrides config).
        max_grad_norm: Max gradient norm for clipping (overrides config).
        target_kl: Target KL divergence for early stopping (overrides config).
        stats_window_size: Window size for rollout statistics (overrides config).
        policy_kwargs: Custom network architecture (overrides config).
        session_id: A unique identifier for the training session.
    """
    # Override default configuration with any provided arguments
    config = TRAINING_CONFIG.copy()
    config.update({k: v for k, v in locals().items() if v is not None and k in config})

    # Create session-specific directories for better organization
    session_name = f"session_{session_id}" if session_id else "default_session"
    session_dir = MODEL_PATHS["best_model"].parent / session_name
    session_checkpoints_dir = session_dir / "checkpoints"
    session_tensorboard_dir = TENSORBOARD_DIR / session_name
    session_logs_dir = LOGS_DIR / session_name

    for path in [session_dir, session_checkpoints_dir, session_tensorboard_dir, session_logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Create the vectorized training and evaluation environments
    train_env = create_vec_env(config["num_envs"], seed)
    eval_env = create_vec_env(1, seed + config["num_envs"]) # Use a different seed for eval

    # Set up callbacks for evaluation and checkpointing
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(session_dir),
        log_path=str(session_logs_dir),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=EVAL_CONFIG["deterministic"],
        render=EVAL_CONFIG["render"]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["eval_freq"],
        save_path=str(session_checkpoints_dir),
        name_prefix=f"ppo_multi_elevator_{session_id or 'model'}"
    )

    # Initialize the PPO model with specified hyperparameters
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        target_kl=config["target_kl"],
        stats_window_size=config["stats_window_size"],
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=str(session_tensorboard_dir),
        verbose=1
    )

    # Start the training process
    try:
        logger.info(f"Starting training session: {session_name}")
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
    finally:
        # Save the final model and clean up
        final_model_path = session_dir / "latest_model.zip"
        model.save(final_model_path)
        logger.info(f"Training complete. Final model saved to {final_model_path}")
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    # Set a fixed random seed for reproducibility of the training process
    seed = 42

    # Start the training process with the specified seed
    train(seed=seed) 