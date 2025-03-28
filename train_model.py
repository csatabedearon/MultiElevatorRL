# train_model.py
import logging  # Import the logging module
import multiprocessing
from multi_elevator.train import train

# --- Add this section to configure logging ---
# Get the specific logger used by the environment module
# The name corresponds to the module path: 'multi_elevator.environment.env'
env_logger = logging.getLogger('multi_elevator.environment.env')

# Set its level to INFO for the training script.
# This means DEBUG messages from the environment logger won't be processed.
env_logger.setLevel(logging.INFO)

# Optional: You might also want to control the level of Stable Baselines3 logs
# logging.getLogger('stable_baselines3').setLevel(logging.INFO)

# Note: The handler defined in train.py will still show INFO messages from env_logger
# and the handler in env.py will also show INFO messages.
# If you ONLY want train.py's handlers active, you could potentially remove
# the handler from env_logger here, but simply setting the level is often sufficient.
# Example (more advanced):
# for h in env_logger.handlers[:]: # Iterate over a copy
#    if isinstance(h, logging.StreamHandler):
#        env_logger.removeHandler(h)
# ---------------------------------------------

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # --- Define the desired policy network architecture ---
    custom_policy_kwargs = {
        "net_arch": dict(
            pi=[128, 128],  # Policy network layers
            vf=[128, 128]   # Value function network layers
        )
    }
    # -------------------------------------------------------

    # --- Define the desired total timesteps ---
    total_training_steps = 3_000_000  # Set to 3 million
    # ------------------------------------------

    # Train with custom policy and total timesteps
    train(
        seed=42,
        policy_kwargs=custom_policy_kwargs,
        total_timesteps=total_training_steps
    )