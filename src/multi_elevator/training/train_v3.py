from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from multi_elevator.environment.env import MultiElevatorEnv
import numpy as np
import torch

# Set global seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def make_env(seed_offset=0):
    def _init():
        env = MultiElevatorEnv(render_mode=None)
        env.reset(seed=seed + seed_offset)
        return env
    return _init

# Custom callback for detailed logging
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self):
        if "episode" in self.locals["infos"][0]:
            reward = self.locals["rewards"][0]
            self.episode_rewards.append(reward)
            print(f"Episode Reward: {reward}, Avg: {sum(self.episode_rewards)/len(self.episode_rewards):.2f}")
        return True

if __name__ == '__main__':
    # Match num_envs to your CPU cores (e.g., 6 for Ryzen 5 5600)
    num_envs = 6
    train_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Consistent evaluation environment
    eval_env = MultiElevatorEnv(render_mode=None)
    eval_env.reset(seed=seed)

    # Evaluation callback with less frequent checks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_v11/",
        log_path="./logs/",
        eval_freq=2_083,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    custom_callback = CustomCallback()

    # PPO with tuned hyperparameters
    model = PPO(
        "MultiInputPolicy",
        train_env,
        n_steps=4096,
        learning_rate=1e-4,
        clip_range=0.1,
        tensorboard_log="./ppo_multi_elevator_tensorboard/",
        verbose=1,
        seed=seed
    )

    # Train with both callbacks
    model.learn(total_timesteps=1_000_000, callback=[eval_callback, custom_callback])

    # Save final model
    model.save("ppo_multi_elevator_v11_01AR.zip")

    # Clean up
    train_env.close()
    eval_env.close()