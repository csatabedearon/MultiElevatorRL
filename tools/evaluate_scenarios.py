"""Evaluate elevator policies on fixed and capacity-stress scenarios."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

import multi_elevator.environment.env as env_module
from multi_elevator.environment.env import MultiElevatorEnv

logging.disable(logging.DEBUG)
ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(8000, 8020))
SCENARIOS = {
    "balanced": [
        {"arrival_step": 1, "start_floor": 0, "destination_floor": 4},
        {"arrival_step": 1, "start_floor": 4, "destination_floor": 0},
        {"arrival_step": 2, "start_floor": 1, "destination_floor": 3},
        {"arrival_step": 2, "start_floor": 3, "destination_floor": 1},
        {"arrival_step": 5, "start_floor": 2, "destination_floor": 0},
        {"arrival_step": 5, "start_floor": 0, "destination_floor": 2},
    ],
    "burst": [
        {"arrival_step": 1, "start_floor": 0, "destination_floor": 4},
        {"arrival_step": 1, "start_floor": 0, "destination_floor": 3},
        {"arrival_step": 1, "start_floor": 1, "destination_floor": 4},
        {"arrival_step": 1, "start_floor": 1, "destination_floor": 0},
        {"arrival_step": 1, "start_floor": 3, "destination_floor": 0},
        {"arrival_step": 1, "start_floor": 4, "destination_floor": 1},
        {"arrival_step": 1, "start_floor": 4, "destination_floor": 2},
        {"arrival_step": 1, "start_floor": 2, "destination_floor": 4},
    ],
    "late_calls": [
        {"arrival_step": 1, "start_floor": 0, "destination_floor": 4},
        {"arrival_step": 30, "start_floor": 4, "destination_floor": 0},
        {"arrival_step": 60, "start_floor": 1, "destination_floor": 3},
        {"arrival_step": 90, "start_floor": 3, "destination_floor": 1},
    ],
    "capacity_burst": [
        *({"arrival_step": 1, "start_floor": 0, "destination_floor": 4} for _ in range(30)),
        *({"arrival_step": 1, "start_floor": 4, "destination_floor": 0} for _ in range(30)),
    ],
}
MODELS = {
    "v15_waiting_ages": ROOT / "models/best_models/session_v15_waiting_ages/latest_model.zip",
    "v18_2m_waiting_ages": ROOT / "models/best_models/session_v18_2m_waiting_ages/latest_model.zip",
    "v20_capacity_2m": ROOT / "models/best_models/session_v20_capacity_2m/latest_model.zip",
    "v21_delivery_weight": ROOT / "models/best_models/session_v21_delivery_weight_0_30/latest_model.zip",
    "v23_optuna_best_2m": ROOT / "models/best_models/session_v23_optuna_best_2m/latest_model.zip",
    "v24_mixed_rate_delivery_2m": ROOT / "models/best_models/session_v24_mixed_rate_delivery_2m/latest_model.zip",
    "v25_mixed_rate_delivery_seed23_2m": ROOT / "models/best_models/session_v25_mixed_rate_delivery_seed23_2m/latest_model.zip",
    "v26_mixed_rate_delivery_seed24_2m": ROOT / "models/best_models/session_v26_mixed_rate_delivery_seed24_2m/latest_model.zip",
}


def run_episode(model, scenario, seed):
    load_space = model.observation_space.spaces.get("elevator_loads")
    env = MultiElevatorEnv(
        max_steps=300,
        passenger_rate=0.05,
        num_elevators=3,
        num_floors=5,
        include_waiting_ages="waiting_ages" in model.observation_space.spaces,
        include_passenger_rate="passenger_rate" in model.observation_space.spaces,
        max_passengers_per_elevator=int(load_space.high[0]) if load_space is not None else None,
        include_elevator_loads=load_space is not None,
        scenario=scenario,
    )
    try:
        obs, _ = env.reset(seed=seed)
        total_reward = 0.0
        movements = 0
        max_load = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            movements += int(np.count_nonzero(action))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            max_load = max(max_load, int(info.get("max_elevator_load", 0)))
        records = list(env.waiting_time.values())
        picked = [p for p in records if p["wait_end"] is not None]
        served = [p for p in records if p["travel_end"] is not None]
        return {
            "reward": total_reward,
            "arrivals": len(records),
            "served": len(served),
            "pickup_wait": float(np.mean([p["wait_end"] - p["wait_start"] for p in picked])) if picked else None,
            "total_wait": float(np.mean([p["travel_end"] - p["wait_start"] for p in served])) if served else None,
            "outstanding": len(records) - len(served),
            "movements": movements,
            "max_load": max_load,
        }
    finally:
        env.close()


results = {}
for label, path in MODELS.items():
    env_module.WEIGHT_DELIVERY = 0.3 if label.startswith(("v21_", "v24_")) else 0.15
    model = PPO.load(str(path), device="cpu")
    results[label] = {}
    for scenario_name, scenario in SCENARIOS.items():
        episodes = [run_episode(model, scenario, seed) for seed in SEEDS]
        summary = {key: float(np.mean([ep[key] for ep in episodes if ep[key] is not None])) for key in episodes[0]}
        summary["service_rate"] = summary["served"] / max(1, summary["arrivals"])
        results[label][scenario_name] = summary

out = ROOT / "logs" / "fixed_scenario_eval.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
print(f"saved={out}")
