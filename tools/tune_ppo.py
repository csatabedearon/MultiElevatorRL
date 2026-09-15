"""Optuna tuner for PPO on the multi-elevator environment."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Keep child environments quiet; the trial/evaluation metrics are the signal.
logging.disable(logging.DEBUG)
torch.set_num_threads(1)

from multi_elevator.config.training_config import TRAINING_CONFIG, TENSORBOARD_DIR
from multi_elevator.environment.env import MultiElevatorEnv
from multi_elevator.train import TrainingMetricsCallback


EVAL_SEEDS = (6000, 6001, 6002, 6003)


def env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_elevators": config["num_elevators"],
        "num_floors": config["num_floors"],
        "max_steps": config["max_steps"],
        "passenger_rate": config["passenger_rate"],
        "include_waiting_ages": config["include_waiting_ages"],
        "max_passengers_per_elevator": config["max_passengers_per_elevator"],
        "include_elevator_loads": config["include_elevator_loads"],
        "passenger_rate_range": config.get("passenger_rate_range"),
        "include_passenger_rate": config.get("include_passenger_rate", False),
    }


def make_trial_vec_env(config: dict[str, Any], seed: int):
    def make_one(rank: int):
        kwargs = env_kwargs(config)
        set_random_seed(seed + rank)

        def init_env():
            env = MultiElevatorEnv(**kwargs)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return init_env

    factories = [make_one(rank) for rank in range(config["num_envs"])]
    if config["num_envs"] > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def run_episode(model: PPO, config: dict[str, Any], seed: int) -> dict[str, float]:
    env = MultiElevatorEnv(**env_kwargs(config))
    try:
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        reward = 0.0
        movements = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            movements += int(np.count_nonzero(action))
            observation, step_reward, terminated, truncated, _ = env.step(action)
            reward += float(step_reward)

        records = list(env.waiting_time.values())
        picked = [record for record in records if record["wait_end"] is not None]
        served = [record for record in records if record["travel_end"] is not None]
        pickup_wait = (
            np.mean([record["wait_end"] - record["wait_start"] for record in picked])
            if picked else 0.0
        )
        total_wait = (
            np.mean([record["travel_end"] - record["wait_start"] for record in served])
            if served else float(config["max_steps"])
        )
        service_rate = len(served) / max(1, len(records))
        return {
            "reward": float(reward),
            "pickup_wait": float(pickup_wait),
            "total_wait_to_dropoff": float(total_wait),
            "service_rate": float(service_rate),
            "movements": float(movements),
            "outstanding": float(len(records) - len(served)),
        }
    finally:
        env.close()


def evaluate(model: PPO, config: dict[str, Any], seeds=EVAL_SEEDS) -> dict[str, float]:
    episodes = [run_episode(model, config, seed) for seed in seeds]
    return {key: float(np.mean([episode[key] for episode in episodes])) for key in episodes[0]}


def domain_score(metrics: dict[str, float]) -> float:
    """Higher is better; keeps throughput, wait and movement in one objective."""
    return (
        100.0 * metrics["service_rate"]
        - metrics["pickup_wait"]
        - 0.35 * metrics["total_wait_to_dropoff"]
        - 0.005 * metrics["movements"]
    )


class TrialEvalCallback(BaseCallback):
    """Evaluate, report to Optuna, and prune weak trials."""

    def __init__(
        self,
        trial: optuna.Trial,
        config: dict[str, Any],
        eval_freq_calls: int,
        eval_seeds: tuple[int, ...],
    ):
        super().__init__(verbose=0)
        self.trial = trial
        self.config = config
        self.eval_freq_calls = max(1, eval_freq_calls)
        self.eval_seeds = eval_seeds
        self.eval_index = 0
        self.history: list[dict[str, float]] = []
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq_calls:
            return True
        metrics = evaluate(self.model, self.config, self.eval_seeds)
        score = domain_score(metrics)
        record = {**metrics, "score": score, "eval_index": float(self.eval_index)}
        self.history.append(record)
        self.logger.record("optuna/domain_score", score)
        self.logger.record("optuna/service_rate", metrics["service_rate"])
        self.logger.record("optuna/pickup_wait", metrics["pickup_wait"])
        self.logger.record("optuna/total_wait_to_dropoff", metrics["total_wait_to_dropoff"])
        self.trial.report(score, self.eval_index)
        self.eval_index += 1
        if self.trial.should_prune():
            self.is_pruned = True
            return False
        return True


def sample_ppo_params(trial: optuna.Trial) -> dict[str, Any]:
    """Small, evidence-based PPO search space from RL Zoo/SB3 practice."""
    n_steps_pow = trial.suggest_int("n_steps_pow", 8, 11)
    activation_name = trial.suggest_categorical("activation", ["tanh", "relu"])
    width = trial.suggest_categorical("net_width", [64, 128])
    target_kl_name = trial.suggest_categorical("target_kl", ["none", "0.01", "0.03"])
    activation = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[activation_name]
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-3, log=True),
        "n_steps": 2 ** n_steps_pow,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10, 15]),
        "gamma": trial.suggest_float("gamma", 0.98, 0.9995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "clip_range": trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.03, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.8),
        "target_kl": None if target_kl_name == "none" else float(target_kl_name),
        "policy_kwargs": {
            "activation_fn": activation,
            "net_arch": {"pi": [width, width], "vf": [width, width]},
        },
    }


def json_params(params: dict[str, Any]) -> dict[str, Any]:
    result = dict(params)
    policy_kwargs = result.get("policy_kwargs")
    if policy_kwargs:
        activation_fn = policy_kwargs["activation_fn"]
        activation_name = getattr(activation_fn, "__name__", str(activation_fn))
        result["policy_kwargs"] = {
            "net_arch": policy_kwargs["net_arch"],
            "activation_fn": activation_name,
        }
    return result


def build_objective(args: argparse.Namespace):
    base_config = dict(TRAINING_CONFIG)
    base_config.update(
        {
            "num_envs": args.n_envs,
            "passenger_rate": args.rate,
        }
    )
    study_tb_dir = TENSORBOARD_DIR / "optuna" / args.study_name
    study_tb_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = tuple(range(6000, 6000 + args.eval_episodes))

    def objective(trial: optuna.Trial) -> float:
        config = dict(base_config)
        ppo_params = sample_ppo_params(trial)
        trial.set_user_attr("ppo_params", json_params(ppo_params))
        trial_dir = study_tb_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        train_env = make_trial_vec_env(config, args.seed + trial.number * 100)
        callback = TrialEvalCallback(
            trial,
            config,
            eval_freq_calls=max(1, args.eval_freq // config["num_envs"]),
            eval_seeds=eval_seeds,
        )
        model = PPO(
            "MultiInputPolicy",
            train_env,
            **ppo_params,
            tensorboard_log=str(trial_dir),
            verbose=0,
            seed=args.seed + trial.number,
        )
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=[callback, TrainingMetricsCallback()],
                tb_log_name=f"trial_{trial.number}",
                progress_bar=False,
            )
            if callback.is_pruned:
                raise optuna.exceptions.TrialPruned()
            if not callback.history:
                callback.history.append({**evaluate(model, config, eval_seeds), "score": 0.0, "eval_index": 0.0})
                callback.history[-1]["score"] = domain_score(callback.history[-1])
            final_metrics = callback.history[-1]
            for key, value in final_metrics.items():
                if key != "eval_index":
                    trial.set_user_attr(key, float(value))
            model.save(trial_dir / "latest_model")
            return float(final_metrics["score"])
        except (AssertionError, ValueError, RuntimeError) as exc:
            trial.set_user_attr("failure", str(exc))
            raise
        finally:
            train_env.close()

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="multi_elevator_ppo_capacity")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=160_000)
    parser.add_argument("--eval-freq", type=int, default=40_000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps({"args": vars(args), "search": "learning_rate/gamma/gae_lambda/n_steps/batch_size/n_epochs/clip_range/ent_coef/vf_coef/activation/net_width/target_kl"}, indent=2))
        return

    db_path = ROOT / "logs" / "optuna_ppo.sqlite3"
    storage = f"sqlite:///{db_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, multivariate=True, seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )
    study.optimize(build_objective(args), n_trials=args.trials, n_jobs=1)

    records = []
    for trial in study.trials:
        records.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    out = ROOT / "logs" / f"{args.study_name}_trials.json"
    out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    best = study.best_trial
    best_out = ROOT / "logs" / f"{args.study_name}_best.json"
    best_out.write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "trial": best.number,
                "value": best.value,
                "params": json_params(best.user_attrs.get("ppo_params", {})) or best.params,
                "raw_params": best.params,
                "user_attrs": best.user_attrs,
                "config": {"rate": args.rate, "n_envs": args.n_envs, "timesteps": args.timesteps},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"best_trial": best.number, "value": best.value, "params": best.params, "saved": str(best_out)}, indent=2, default=str))


if __name__ == "__main__":
    main()
