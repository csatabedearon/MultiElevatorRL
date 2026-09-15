import logging

import numpy as np

from multi_elevator.environment.env import MultiElevatorEnv


logging.getLogger("multi_elevator.environment.env").setLevel(logging.WARNING)


def test_max_steps_is_reported_as_truncation():
    env = MultiElevatorEnv(
        max_steps=1,
        passenger_rate=0.0,
        num_elevators=1,
        num_floors=2,
    )
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)

        _, _, terminated, truncated, _ = env.step(np.array([0], dtype=np.int64))

        assert terminated is False
        assert truncated is True
    finally:
        env.close()


def test_observation_exposes_waiting_ages():
    env = MultiElevatorEnv(
        max_steps=10,
        passenger_rate=0.0,
        num_elevators=1,
        num_floors=3,
        include_waiting_ages=True,
        scenario=[
            {"arrival_step": 1, "start_floor": 0, "destination_floor": 2},
        ],
    )
    try:
        observation, _ = env.reset(seed=7)
        assert observation["waiting_ages"].tolist() == [0, 0, 0]

        observation, _, _, _, _ = env.step(np.array([0], dtype=np.int64))
        assert observation["waiting_ages"].tolist() == [0, 0, 0]

        observation, _, _, _, _ = env.step(np.array([1], dtype=np.int64))
        assert observation["waiting_ages"].tolist() == [1, 0, 0]
    finally:
        env.close()


def test_waiting_ages_can_be_disabled_for_legacy_models():
    env = MultiElevatorEnv(
        max_steps=5,
        passenger_rate=0.0,
        num_elevators=1,
        num_floors=3,
    )
    try:
        assert "waiting_ages" not in env.observation_space.spaces
        observation, _ = env.reset(seed=7)
        assert "waiting_ages" not in observation
    finally:
        env.close()


def test_training_env_uses_waiting_age_observation():
    from multi_elevator.train import make_env

    env = make_env(0, seed=7)()
    try:
        assert "waiting_ages" in env.observation_space.spaces
    finally:
        env.close()


def test_training_metrics_are_written_to_tensorboard(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from multi_elevator.train import TrainingMetricsCallback

    env = DummyVecEnv([
        lambda: MultiElevatorEnv(
            max_steps=16,
            passenger_rate=0.1,
            num_elevators=3,
            num_floors=5,
            include_waiting_ages=True,
        )
    ])
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=32,
            batch_size=32,
            n_epochs=1,
            tensorboard_log=str(tmp_path),
            seed=7,
            verbose=0,
        )
        model.learn(total_timesteps=64, callback=TrainingMetricsCallback())
    finally:
        env.close()

    event_files = list(tmp_path.rglob("events.out.tfevents.*"))
    assert event_files
    tags = EventAccumulator(str(event_files[0])).Reload().Tags()["scalars"]
    assert "rollout/waiting_passengers" in tags
    assert "rollout/served_passengers" in tags
    assert "rollout/max_elevator_load" in tags
    assert "rollout/passenger_rate" in tags


def test_web_environment_matches_loaded_model_observation():
    import subprocess
    import sys
    from pathlib import Path

    script = """
from stable_baselines3 import PPO
from multi_elevator.utils import web_app

model = PPO.load(
    "models/best_models/session_v15_waiting_ages/latest_model.zip",
    device="cpu",
)
web_app.model = model
web_app.using_model = True
web_app.num_floors = 10
web_app.num_elevators = 3
web_app.include_waiting_ages = False
try:
    web_app.adjust_environment_for_model()
    assert web_app.env.include_waiting_ages is True
    assert "waiting_ages" in web_app.env.observation_space.spaces
finally:
    if web_app.env is not None:
        web_app.env.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_web_environment_matches_capacity_model_observation():
    import subprocess
    import sys
    from pathlib import Path

    script = """
from stable_baselines3 import PPO
from multi_elevator.environment.env import MultiElevatorEnv
from multi_elevator.utils import web_app

source_env = MultiElevatorEnv(
    max_steps=8,
    passenger_rate=0.1,
    num_elevators=3,
    num_floors=5,
    include_waiting_ages=True,
    max_passengers_per_elevator=2,
    include_elevator_loads=True,
)
model = PPO("MultiInputPolicy", source_env, n_steps=8, batch_size=8, n_epochs=1, verbose=0)
web_app.model = model
web_app.using_model = True
web_app.num_floors = 10
web_app.num_elevators = 3
web_app.include_waiting_ages = False
web_app.max_passengers_per_elevator = None
web_app.include_elevator_loads = False
try:
    web_app.adjust_environment_for_model()
    assert web_app.env.include_waiting_ages is True
    assert web_app.env.include_elevator_loads is True
    assert web_app.env.max_passengers_per_elevator == 2
finally:
    if web_app.env is not None:
        web_app.env.close()
    source_env.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_elevator_capacity_limits_pickup():
    env = MultiElevatorEnv(
        max_steps=10,
        passenger_rate=0.0,
        num_elevators=1,
        num_floors=3,
        max_passengers_per_elevator=1,
        include_elevator_loads=True,
        scenario=[
            {"arrival_step": 1, "start_floor": 0, "destination_floor": 2},
            {"arrival_step": 1, "start_floor": 0, "destination_floor": 1},
        ],
    )
    try:
        observation, _ = env.reset(seed=7)
        observation, _, _, _, _ = env.step(np.array([0], dtype=np.int64))
        assert observation["waiting_passengers"].tolist() == [2, 0, 0]

        observation, _, _, _, info = env.step(np.array([0], dtype=np.int64))
        assert observation["elevator_loads"].tolist() == [1]
        assert info["max_elevator_load"] == 1
        assert observation["waiting_passengers"].tolist() == [1, 0, 0]
        assert observation["floor_buttons"].tolist() == [True, False, False]
    finally:
        env.close()


def test_training_env_uses_capacity_and_load_observation():
    from multi_elevator.train import make_env

    env = make_env(0, seed=7)()
    try:
        assert env.unwrapped.max_passengers_per_elevator == 8
        assert "elevator_loads" in env.observation_space.spaces
    finally:
        env.close()


def test_make_env_uses_explicit_environment_config():
    from multi_elevator.train import make_env

    config = {
        "num_elevators": 1,
        "num_floors": 3,
        "max_steps": 10,
        "passenger_rate": 0.0,
        "include_waiting_ages": True,
        "max_passengers_per_elevator": 2,
        "include_elevator_loads": True,
    }
    env = make_env(0, seed=7, config=config)()
    try:
        assert env.unwrapped.num_floors == 3
        assert env.unwrapped.max_passengers_per_elevator == 2
        assert "elevator_loads" in env.observation_space.spaces
    finally:
        env.close()


def test_passenger_rate_range_randomizes_and_is_observed():
    env = MultiElevatorEnv(
        max_steps=10,
        passenger_rate=0.1,
        passenger_rate_range=(0.05, 0.5),
        include_passenger_rate=True,
        num_elevators=1,
        num_floors=3,
    )
    try:
        first, _ = env.reset(seed=1)
        first_rate = float(first["passenger_rate"][0])
        second, _ = env.reset(seed=2)
        second_rate = float(second["passenger_rate"][0])
        assert env.observation_space.contains(first)
        assert env.observation_space.contains(second)
        assert 0.05 <= first_rate <= 0.5
        assert 0.05 <= second_rate <= 0.5
        assert first_rate != second_rate
    finally:
        env.close()
