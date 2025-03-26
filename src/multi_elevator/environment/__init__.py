# src/multi_elevator/environment/__init__.py
"""
Multi-Elevator environment package.
"""
# First import the environment class
from .env import MultiElevatorEnv  # This imports env.py

# Then import the registration function
from gymnasium.envs.registration import register

# Perform the registration
register(
    id='MultiElevator-v0',
    entry_point='multi_elevator.environment:MultiElevatorEnv', # Points to the class
    max_episode_steps=500,  # Default max steps per episode
)

# Make the class available when importing the package
__all__ = ['MultiElevatorEnv']