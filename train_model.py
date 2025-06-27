"""
This script serves as the main entry point for training the Multi-Elevator RL model.

It configures the training process with specific hyperparameters and initiates the
training loop defined in `multi_elevator.train`.

This script demonstrates how to:
- Set up logging to control the verbosity of the environment and training logs.
- Define a custom neural network architecture for the policy and value functions.
- Specify the total number of timesteps for the training session.
- Assign a unique session ID for organized logging and model saving.
"""
import logging
import multiprocessing
import datetime

from multi_elevator.train import train

# --- Logging Configuration ---
# The environment (`env.py`) uses a logger named 'multi_elevator.environment.env'.
# By default, it's set to DEBUG. For training, we often don't need such verbose
# output from the environment. Here, we get that specific logger and set its
# level to INFO, so only messages of INFO level or higher (INFO, WARNING, ERROR)
# will be processed by the handlers defined in the training script.
env_logger = logging.getLogger('multi_elevator.environment.env')
env_logger.setLevel(logging.INFO)

if __name__ == '__main__':
    # Ensure compatibility with Windows multiprocessing
    multiprocessing.freeze_support()

    # --- Session Configuration ---
    # Generate a unique timestamp to serve as a session ID.
    # This helps in organizing models and logs from different training runs.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Hyperparameter and Architecture Definition ---
    # Define a custom network architecture for the PPO agent.
    # 'pi' (policy) and 'vf' (value function) networks can have different structures.
    custom_policy_kwargs = {
        "net_arch": {
            "pi": [128, 128],  # Policy network with two hidden layers of 128 neurons each
            "vf": [128, 128]   # Value network with two hidden layers of 128 neurons each
        }
    }

    # Define the total number of timesteps for this training run.
    total_training_steps = 5_000_000

    # --- Start Training ---
    # Call the main training function with the specified configuration.
    train(
        seed=22,  # Use a fixed seed for reproducibility
        policy_kwargs=custom_policy_kwargs,
        total_timesteps=total_training_steps,
        session_id=timestamp  # Pass the unique ID for this session
    )