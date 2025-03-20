from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_elevator_env import MultiElevatorEnv

def make_env(seed_offset=0):
    def _init():
        env = MultiElevatorEnv(render_mode=None)
        env.reset(seed=42 + seed_offset)
        return env
    return _init

# Create vectorized training environment with multiple parallel environments
num_envs = 12  # Adjust this based on your available cores/resources
train_env = DummyVecEnv([make_env(i) for i in range(num_envs)])

# Create evaluation environment (using a single instance is usually sufficient)
eval_env = MultiElevatorEnv(render_mode=None)

# Set up evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./best_model_v11/",
    log_path="./logs/",
    eval_freq=2_083,  # Evaluate every 10,000 steps
    deterministic=True,
    render=False
)

# Create PPO model with the vectorized training environment
model = PPO(
    "MultiInputPolicy",
    train_env,
    tensorboard_log="./ppo_multi_elevator_tensorboard/",
    verbose=1,
)

# Train the model with the evaluation callback
model.learn(total_timesteps=1_000_000, callback=eval_callback)

# Save the final model
model.save("ppo_multi_elevator_v11_03AR.zip")
