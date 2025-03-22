from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from multi_elevator.environment.env import MultiElevatorEnv

# Create environment
env = MultiElevatorEnv(render_mode=None)
env.reset(seed=42)

# Create model
model = PPO(
    "MultiInputPolicy",
    env,
    tensorboard_log="./ppo_multi_elevator_tensorboard/",
    verbose=1,
)

# Set up evaluation environment
eval_env = MultiElevatorEnv(render_mode=None)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10_000,  # Evaluate every 10,000 steps
    deterministic=True,
    render=False
)

# Train the model with the callback
model.learn(total_timesteps=1_000_000, callback=eval_callback)

# Save final model
model.save("ppo_multi_elevator_final.zip")
