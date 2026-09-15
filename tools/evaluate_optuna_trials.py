"""Re-evaluate the top Optuna trial checkpoints on fixed traffic seeds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_elevator.config.training_config import TRAINING_CONFIG
from tools.tune_ppo import domain_score, run_episode

SEEDS = tuple(range(7000, 7020))
RATES = (0.1, 0.3, 0.5)
STUDY = "ppo_capacity_tuning_v1"

trials = json.loads((ROOT / "logs" / f"{STUDY}_trials.json").read_text(encoding="utf-8"))
complete = sorted(
    [trial for trial in trials if trial["state"] == "COMPLETE" and trial["value"] is not None],
    key=lambda trial: trial["value"],
    reverse=True,
)[:3]

results = []
for trial in complete:
    path = ROOT / "logs" / "tensorboard" / "optuna" / STUDY / f"trial_{trial['number']}" / "latest_model.zip"
    model = PPO.load(str(path), device="cpu")
    trial_result = {"trial": trial["number"], "pilot_score": trial["value"], "path": str(path), "rates": []}
    for rate in RATES:
        config = dict(TRAINING_CONFIG)
        config["passenger_rate"] = rate
        episodes = [run_episode(model, config, seed) for seed in SEEDS]
        metrics = {key: float(np.mean([episode[key] for episode in episodes])) for key in episodes[0]}
        metrics.update({"rate": rate, "score": domain_score(metrics), "episodes": len(episodes)})
        trial_result["rates"].append(metrics)
    results.append(trial_result)

out = ROOT / "logs" / f"{STUDY}_top3_eval.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"saved={out}")
