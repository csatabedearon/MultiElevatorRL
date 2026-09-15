"""Train a capacity-enabled PPO policy with episode-level traffic randomization."""
from __future__ import annotations

import argparse
import logging
import multiprocessing

logging.disable(logging.DEBUG)

from multi_elevator.config import training_config
from multi_elevator.environment import env as env_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--rate-low", type=float, default=0.05)
    parser.add_argument("--rate-high", type=float, default=0.50)
    parser.add_argument("--delivery-weight", type=float, default=0.30)
    parser.add_argument("--session-id", default="mixed_rate_delivery_2m")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.rate_low <= args.rate_high <= 1.0:
        raise ValueError("rate range must satisfy 0 <= low <= high <= 1")
    training_config.TRAINING_CONFIG["passenger_rate_range"] = (args.rate_low, args.rate_high)
    training_config.TRAINING_CONFIG["include_passenger_rate"] = True
    env_module.WEIGHT_DELIVERY = args.delivery_weight

    from multi_elevator.train import train

    train(
        seed=args.seed,
        total_timesteps=args.timesteps,
        session_id=args.session_id,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
