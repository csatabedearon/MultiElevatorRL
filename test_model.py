from multi_elevator_env import MultiElevatorEnv
from stable_baselines3 import PPO

# Create the environment (with rendering for visualization)
env = MultiElevatorEnv(render_mode="human")

# Load the trained model
model = PPO.load("ppo_multi_elevator_v9_01AR.zip", env=env)

# Reset the environment with an optional seed for reproducibility
obs, info = env.reset(seed=44)

# Initialize counters for episode statistics
episode_reward = 0
total_moves = 0
episode = 1

# Run the model for a fixed number of steps
for _ in range(5000):
    # Predict action using the loaded model (deterministic for best action)
    action, _states = model.predict(obs, deterministic=True)
    
    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Accumulate reward
    episode_reward += reward

    # Count as a movement if any elevator has a non-idle action
    if any(a != 0 for a in action):
        total_moves += 1
    
    # Reset if the episode has ended
    if terminated or truncated:
        # The info dictionary contains the average waiting time calculated as:
        # total_waiting_time / max(1, total_passengers_served)
        avg_wait_time = info.get("average_waiting_time", 0)
        total_served= info.get("total_passengers_served",0)
        print(f"Episode {episode} finished. Total reward: {episode_reward:.2f}, Total movements: {total_moves}, Average waiting time: {avg_wait_time:.2f} steps, Total passangers served: {total_served:.2f}")
        
        # Reset for the next episode
        obs, info = env.reset()
        episode_reward = 0
        total_moves = 0
        episode += 1

# Close the environment
env.close()
