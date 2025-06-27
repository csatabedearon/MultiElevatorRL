"""
This package handles the registration of the custom Multi-Elevator environment
with the Gymnasium library.

By importing this package, the `MultiElevator-v0` environment becomes available
to be created via `gymnasium.make()`.

It imports the `MultiElevatorEnv` class and uses `gymnasium.register` to make it
accessible by its ID.
"""
# First, import the environment class to make it available.
from .env import MultiElevatorEnv

# Then, import the registration function from Gymnasium.
from gymnasium.envs.registration import register

# Perform the registration of the custom environment.
register(
    id='MultiElevator-v0',                                 # Unique ID for the environment
    entry_point='multi_elevator.environment:MultiElevatorEnv', # Path to the environment class
    max_episode_steps=500,                                # Default max steps per episode
)

# Make the environment class directly accessible when importing the package.
__all__ = ['MultiElevatorEnv']