# Codebase patterns

- Python package code lives under `src/multi_elevator`; tests use `pytest`.
- `MultiElevatorEnv` exposes a Gymnasium dict observation and `MultiDiscrete` elevator actions.
- Training is orchestrated by `multi_elevator.train.train`; paths and defaults live in `config/training_config.py`.
- Saved checkpoints are not configuration-compatible: older models use 10 floors, `v13` uses 5 floors.
- Environment time limits must be returned as `truncated=True`, not `terminated=True`.
- New checkpoints may add `waiting_ages`, `elevator_loads`, or `passenger_rate`; infer observation keys from the loaded model before evaluation or webapp startup.
- `make_env` and `create_vec_env` accept an explicit config snapshot; do not rely on the mutable global training config inside workers.
- Capacity-enabled training uses `max_passengers_per_elevator=8` and must be benchmarked separately from legacy unlimited-capacity models.
- Optuna tuning lives in `tools/tune_ppo.py`: SQLite persistence, TPE sampling, intermediate domain evaluation, and MedianPruner.
- `VecNormalize` may normalize only Box observation keys here (`elevator_loads`, `waiting_ages`, `waiting_passengers`); save and reload its statistics for evaluation.
- VecEnv evaluation auto-resets on terminal steps; inspect terminal state before the reset or use a non-terminal evaluation horizon.
