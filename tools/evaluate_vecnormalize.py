"""Evaluate a VecNormalize-trained policy with its saved running statistics."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from multi_elevator.config.training_config import TRAINING_CONFIG
from multi_elevator.train import create_vec_env

logging.disable(logging.DEBUG)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models/best_models/session_v22_vecnormalize_obs_2m"
SEEDS = tuple(range(9000, 9020))
RATES = (0.1, 0.3, 0.5)


def run(model, vec, seed):
    vec.seed(seed)
    obs = vec.reset()
    reward = 0.0
    movements = 0
    max_load = 0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        movements += int(np.count_nonzero(action))
        obs, rewards, dones, infos = vec.step(action)
        reward += float(rewards[0])
        max_load = max(max_load, int(infos[0].get("max_elevator_load", 0)))
    env = vec.venv.envs[0].unwrapped
    records = list(env.waiting_time.values())
    picked = [p for p in records if p["wait_end"] is not None]
    served = [p for p in records if p["travel_end"] is not None]
    return {
        "reward": reward,
        "arrivals": len(records),
        "served": len(served),
        "pickup_wait": float(np.mean([p["wait_end"] - p["wait_start"] for p in picked])) if picked else 0.0,
        "total_wait": float(np.mean([p["travel_end"] - p["wait_start"] for p in served])) if served else 500.0,
        "outstanding": len(records) - len(served),
        "movements": movements,
        "max_load": max_load,
    }


results = []
for rate in RATES:
    config = dict(TRAINING_CONFIG)
    config["passenger_rate"] = rate
    config["max_steps"] = 501
    raw = create_vec_env(1, 9000, config)
    vec = VecNormalize.load(str(MODEL_DIR / "vecnormalize.pkl"), raw)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(str(MODEL_DIR / "latest_model.zip"), env=vec, device="cpu")
    try:
        episodes = [run(model, vec, seed) for seed in SEEDS]
        summary = {key: float(np.mean([ep[key] for ep in episodes])) for key in episodes[0]}
        summary.update({"rate": rate, "service_rate": summary["served"] / max(1.0, summary["arrivals"])})
        results.append(summary)
    finally:
        vec.close()

out = ROOT / "logs" / "v22_vecnormalize_rate_eval.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"saved={out}")
