"""Train PPO with VecNormalize on the numeric observation fields."""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from multi_elevator.config.training_config import (
    EVAL_CONFIG,
    LOGS_DIR,
    MODEL_PATHS,
    TENSORBOARD_DIR,
    TRAINING_CONFIG,
)
from multi_elevator.train import TrainingMetricsCallback, create_vec_env

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_KEYS = ["elevator_loads", "waiting_ages", "waiting_passengers"]
logging.disable(logging.DEBUG)
torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--session-id", default="v22_vecnormalize_obs_2m")
    parser.add_argument("--norm-reward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(TRAINING_CONFIG)
    session_name = f"session_{args.session_id}"
    session_dir = MODEL_PATHS["best_model"].parent / session_name
    checkpoint_dir = session_dir / "checkpoints"
    tensorboard_dir = TENSORBOARD_DIR / session_name
    log_dir = LOGS_DIR / session_name
    for directory in [session_dir, checkpoint_dir, tensorboard_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_raw = create_vec_env(config["num_envs"], args.seed, config)
    train_env = VecNormalize(
        train_raw,
        norm_obs=True,
        norm_obs_keys=NORMALIZED_KEYS,
        norm_reward=args.norm_reward,
        clip_obs=10.0,
    )
    eval_raw = create_vec_env(1, args.seed + config["num_envs"], config)
    eval_env = VecNormalize(
        eval_raw,
        norm_obs=True,
        norm_obs_keys=NORMALIZED_KEYS,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(session_dir),
        log_path=str(log_dir),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=EVAL_CONFIG["deterministic"],
        render=EVAL_CONFIG["render"],
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config["eval_freq"],
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_multi_elevator_{args.session_id}",
    )
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
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
        seed=args.seed,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_callback, checkpoint_callback, TrainingMetricsCallback()],
            progress_bar=True,
        )
    finally:
        model.save(session_dir / "latest_model")
        train_env.save(str(session_dir / "vecnormalize.pkl"))
        train_env.close()
        eval_env.close()
    print(json.dumps({
        "session": session_name,
        "model": str(session_dir / "latest_model.zip"),
        "vecnormalize": str(session_dir / "vecnormalize.pkl"),
        "normalized_keys": NORMALIZED_KEYS,
        "norm_reward": args.norm_reward,
    }, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
