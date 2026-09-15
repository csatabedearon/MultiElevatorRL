# MultiElevatorRL Improvement Report

Date: 2026-09-15

## Scope

This report records the environment, reward, PPO, evaluation, and tooling improvements made after cloning the original state-exam project. Generated model archives and TensorBoard/Optuna databases remain local and are intentionally ignored by Git; the reproducible code and this report are committed.

## Code changes

- Corrected Gymnasium time-limit semantics: time limits return `truncated=True`, not `terminated=True`.
- Added optional `waiting_ages` observations.
- Added optional elevator capacity enforcement (`max_passengers_per_elevator`) and `elevator_loads` observations.
- Added `max_elevator_load` and `passenger_rate` TensorBoard metrics.
- Made environment configuration explicit in `make_env`/`create_vec_env`; worker processes no longer silently read stale global configuration.
- Updated the web application to infer waiting-age/load/capacity compatibility from a loaded model.
- Added optional episode-level traffic randomization with `passenger_rate_range` and an observed `passenger_rate` scalar.
- Added Optuna tooling with SQLite persistence, TPE sampling, intermediate domain evaluation, and Median pruning.
- Added a reproducible mixed-rate trainer and a `VecNormalize` trainer/evaluator.
- Added regression and compatibility tests.

## Verification

- Full test suite: **11 passed**.
- Python compilation: passed for source and experiment tools.
- `git diff --check`: passed.
- Stable-Baselines3 Gymnasium environment checks and smoke trainings passed during the improvement cycle.
- TensorBoard: `http://127.0.0.1:6006/`.
- Optuna Dashboard: `http://127.0.0.1:6007/`.

## Experiments

All comparisons used compatible 5-floor/3-elevator environments and fixed evaluation seeds unless noted otherwise.

### Baselines and capacity

| Session | Main change | Result |
|---|---|---|
| `session_v15_waiting_ages` | 1M, waiting ages | Strong legacy baseline |
| `session_v18_2m_waiting_ages` | 2M, no capacity | Pickup wait improved over v15; failed at extreme unseen traffic |
| `session_v19_capacity` | 1M, capacity 8 | More realistic but undertrained |
| `session_v20_capacity_2m` | 2M, capacity 8 | Strong efficiency/default baseline |
| `session_v21_delivery_weight_0_30` | 1M, delivery weight 0.30 | Lower total wait than v20, more movement |

At `passenger_rate=0.1`, the fixed-seed benchmark was:

| Model | Pickup wait | Total wait to dropoff | Service | Movements |
|---|---:|---:|---:|---:|
| v20 capacity + 2M | 1.575 | 4.858 | 98.74% | 259.5 |
| v21 delivery weight | 1.730 | **4.294** | 99.01% | 324.4 |
| v23 Optuna + 2M | **1.537** | 4.589 | **99.10%** | 711.1 |

Reward values are not compared between v20 and v21 because v21 intentionally changes the delivery reward weight.

### Optuna

The study `ppo_capacity_tuning_v1` ran 12 trials: 7 completed and 5 were pruned. The best short-budget trial used:

```text
learning_rate = 0.00039923
n_steps = 512
batch_size = 64
n_epochs = 10
gamma = 0.984945
gae_lambda = 0.925715
clip_range = 0.3
ent_coef = 0.001964
vf_coef = 0.3685
activation = ReLU
net_arch = 64/64
target_kl = None
```

The resulting `session_v23_optuna_best_2m` model improved pickup wait but did not beat the mixed-rate models on total wait or movement efficiency.

### Mixed-rate training

The mixed-rate recipe is:

```text
passenger_rate_range = (0.05, 0.50)
include_passenger_rate = True
max_passengers_per_elevator = 8
include_waiting_ages = True
include_elevator_loads = True
WEIGHT_DELIVERY = 0.30
```

Three independent 2M models were trained:

- `session_v24_mixed_rate_delivery_2m` — seed 22
- `session_v25_mixed_rate_delivery_seed23_2m` — seed 23
- `session_v26_mixed_rate_delivery_seed24_2m` — seed 24

Across those three models, 20 fixed evaluation seeds per rate produced:

| Evaluation rate | Avg pickup wait | Avg total wait | Avg service | Worst service |
|---:|---:|---:|---:|---:|
| 0.02 | 1.902 | 4.788 | 99.52% | 99.29% |
| 0.05 | 1.703 | 4.326 | 99.08% | 98.97% |
| 0.10 | 1.710 | 4.137 | 98.98% | 98.74% |
| 0.20 | 1.674 | **4.048** | 99.21% | 99.15% |
| 0.30 | 1.630 | 4.070 | **99.36%** | 99.33% |
| 0.50 | **1.629** | 4.277 | 99.13% | 99.07% |

The mixed-rate policy is the strongest general-purpose family found so far: it remains robust at `0.50` instead of failing out-of-distribution like the fixed-rate v18 model.

### Fixed stress scenarios

All three mixed-rate models served 100% of passengers in the balanced, burst, late-call, and 60-passenger capacity-burst scenarios. Across the three seeds, the capacity-burst average was:

```text
pickup wait: 7.289
arrival-to-dropoff wait: 13.289
service rate: 100%
```

The capacity observation never allowed the trained policy to exceed the configured 8-passenger elevator limit.

### Observation normalization

`VecNormalize` was tested correctly on the Box fields only:

```text
elevator_loads
waiting_ages
waiting_passengers
```

The MultiBinary/MultiDiscrete fields are intentionally not normalized. The 2M normalized model completed, but its corrected evaluation was worse than the raw mixed-rate family (`total wait 9.60` at rate 0.1), so it was not promoted.

### Longer 5M run

`session_v27_mixed_rate_delivery_seed23_5m` completed successfully at `5,013,504` timesteps. Its final rate/scenario benchmark was not completed before this checkpoint, so it is recorded as an unvalidated experimental artifact rather than promoted as the default model.

## Reproduction commands

Install experiment dependencies through the package metadata:

```bash
.venv/Scripts/python.exe -m pip install -e .
```

Run the Optuna pilot:

```bash
.venv/Scripts/python.exe tools/tune_ppo.py \
  --study-name ppo_capacity_tuning_v1 \
  --trials 12 \
  --timesteps 160000 \
  --eval-freq 40000 \
  --eval-episodes 4 \
  --n-envs 8 \
  --rate 0.1
```

Run a mixed-rate training:

```bash
.venv/Scripts/python.exe tools/train_mixed_rate.py \
  --seed 23 \
  --timesteps 2000000 \
  --rate-low 0.05 \
  --rate-high 0.50 \
  --delivery-weight 0.30 \
  --session-id mixed_rate_delivery_2m
```

Start the visual dashboards:

```bash
.venv/Scripts/tensorboard.exe --logdir logs/tensorboard --host 127.0.0.1 --port 6006
.venv/Scripts/optuna-dashboard.exe \
  sqlite:///C:/Users/Treete/MultiElevatorRL/logs/optuna_ppo.sqlite3 \
  --host 127.0.0.1 --port 6007
```

## Decision

The best validated general-purpose family is **mixed-rate + capacity + waiting ages + delivery weight 0.30**, with seed 23 (`v25`) the strongest individual 2M candidate in the fixed evaluations. The 5M seed-23 run exists but must be benchmarked before replacing v25.

## Sources

[1] RL Baselines3 Zoo, Hyperparameter Tuning: https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html

[2] Stable-Baselines3 PPO documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

[3] Antonin Raffin, Automatic Hyperparameter Tuning in Practice: https://araffin.github.io/post/optuna

[4] Novel RL Approach for Efficient Elevator Group Control Systems: https://arxiv.org/html/2507.00011v1

[5] Crites & Barto, Improving Elevator Performance Using Reinforcement Learning: https://papers.neurips.cc/paper_files/paper/1995/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html
