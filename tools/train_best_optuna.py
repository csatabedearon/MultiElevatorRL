"""Train a long-run PPO model from an Optuna best-trial JSON."""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from multi_elevator.config.training_config import (
    EVAL_CONFIG,
    LOGS_DIR,
    MODEL_PATHS,
    TENSORBOARD_DIR,
    TRAINING_CONFIG,
)
from multi_elevator.train import TrainingMetricsCallback, create_vec_env

ROOT = Path(__file__).resolve().parents[1]
logging.disable(logging.DEBUG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="ppo_capacity_tuning_v1")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--session-id", default="v23_optuna_best_2m")
    return parser.parse_args()


def load_ppo_params(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = dict(payload["params"])
    activation = {"Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU}.get(
        params.get("policy_kwargs", {}).get("activation_fn", "Tanh"),
        torch.nn.Tanh,
    )
    params["policy_kwargs"] = {
        "activation_fn": activation,
        "net_arch": params["policy_kwargs"]["net_arch"],
    }
    return params


def main() -> None:
    args = parse_args()
    best_path = ROOT / "logs" / f"{args.study_name}_best.json"
    params = load_ppo_params(best_path)
    config = dict(TRAINING_CONFIG)
    config["total_timesteps"] = args.timesteps
    set_random_seed(args.seed)

    session_name = f"session_{args.session_id}"
    session_dir = MODEL_PATHS["best_model"].parent / session_name
    checkpoint_dir = session_dir / "checkpoints"
    tensorboard_dir = TENSORBOARD_DIR / session_name
    log_dir = LOGS_DIR / session_name
    for directory in [session_dir, checkpoint_dir, tensorboard_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_env = create_vec_env(config["num_envs"], args.seed, config)
    eval_env = create_vec_env(1, args.seed + config["num_envs"], config)
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
        **params,
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
        train_env.close()
        eval_env.close()
    print(json.dumps({"session": session_name, "model": str(session_dir / "latest_model.zip"), "params": params}, default=str, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
