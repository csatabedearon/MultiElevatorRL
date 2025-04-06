"""
Training script for the Multi-Elevator RL model using PPO.
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def make_env(rank: int, seed: int = 0) -> gym.Env:
    """
    Create a wrapped, monitored environment.
    
    Args:
        rank: The rank of the environment
        seed: The seed for the environment
        
    Returns:
        The wrapped environment
    """
    def _init():
        env = MultiElevatorEnv(
            num_elevators=TRAINING_CONFIG["num_elevators"],
            num_floors=TRAINING_CONFIG["num_floors"],
            max_steps=TRAINING_CONFIG["max_steps"],
            passenger_rate=TRAINING_CONFIG["passenger_rate"]
        )
        env = Monitor(env, str(LOGS_DIR / f"env_{rank}"))
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init

def create_vec_env(num_envs: int, seed: int = 0) -> gym.Env:
    """
    Create a vectorized environment.
    
    Args:
        num_envs: Number of environments to create
        seed: The seed for the environments
        
    Returns:
        The vectorized environment
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
    Train the Multi-Elevator RL model.
    
    Args:
        seed: Random seed for reproducibility
        num_envs: Number of environments to use (overrides config)
        total_timesteps: Total number of timesteps to train (overrides config)
        eval_freq: Frequency of evaluation (overrides config)
        n_eval_episodes: Number of episodes to evaluate (overrides config)
        learning_rate: Learning rate (overrides config)
        n_steps: Number of steps per update (overrides config)
        batch_size: Batch size (overrides config)
        n_epochs: Number of epochs per update (overrides config)
        gamma: Discount factor (overrides config)
        gae_lambda: GAE lambda parameter (overrides config)
        clip_range: PPO clip range (overrides config)
        ent_coef: Entropy coefficient (overrides config)
        vf_coef: Value function coefficient (overrides config)
        max_grad_norm: Maximum gradient norm (overrides config)
        target_kl: Target KL divergence (overrides config)
        stats_window_size: Window size for statistics (overrides config)
        policy_kwargs: Policy network architecture (overrides config)
        session_id: Unique identifier for this training session
    """
    # Update config with provided parameters
    config = TRAINING_CONFIG.copy()
    if num_envs is not None:
        config["num_envs"] = num_envs
    if total_timesteps is not None:
        config["total_timesteps"] = total_timesteps
    if eval_freq is not None:
        config["eval_freq"] = eval_freq
    if n_eval_episodes is not None:
        config["n_eval_episodes"] = n_eval_episodes
    
    # Create session-specific directories
    session_dir = MODEL_PATHS["best_model"].parent / f"session_{session_id}" if session_id else MODEL_PATHS["best_model"].parent
    session_dir.mkdir(parents=True, exist_ok=True)
    
    session_checkpoints_dir = session_dir / "checkpoints"
    session_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    session_tensorboard_dir = TENSORBOARD_DIR / f"session_{session_id}" if session_id else TENSORBOARD_DIR
    session_tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    session_logs_dir = LOGS_DIR / f"session_{session_id}" if session_id else LOGS_DIR
    session_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training environment
    train_env = create_vec_env(config["num_envs"], seed)
    
    # Create evaluation environment
    eval_env = create_vec_env(1, seed + config["num_envs"])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(session_dir),
        log_path=str(session_logs_dir),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config["eval_freq"],
        save_path=str(session_checkpoints_dir),
        name_prefix=f"ppo_multi_elevator_{session_id}" if session_id else "ppo_multi_elevator"
    )
    
    # Create and train the model
    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=learning_rate or config["learning_rate"],
        n_steps=n_steps or config["n_steps"],
        batch_size=batch_size or config["batch_size"],
        n_epochs=n_epochs or config["n_epochs"],
        gamma=gamma or config["gamma"],
        gae_lambda=gae_lambda or config["gae_lambda"],
        clip_range=clip_range or config["clip_range"],
        clip_range_vf=config["clip_range_vf"],
        ent_coef=ent_coef or config["ent_coef"],
        vf_coef=vf_coef or config["vf_coef"],
        max_grad_norm=max_grad_norm or config["max_grad_norm"],
        target_kl=target_kl or config["target_kl"],
        stats_window_size=stats_window_size or config["stats_window_size"],
        policy_kwargs=policy_kwargs or config["policy_kwargs"],
        tensorboard_log=str(session_tensorboard_dir),
        verbose=1
    )
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save the final model
        model.save(session_dir / "latest_model.zip")
        logger.info(f"Training completed. Models saved in {session_dir}")
        
        # Close environments
        train_env.close()
        eval_env.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    train(seed=seed) 