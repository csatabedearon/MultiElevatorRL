"""
Constants and configuration values for the MultiElevator environment.

This file centralizes all static parameters used in the simulation,
including environment settings, reward weights, and rendering properties.
"""

# --- Environment Configuration ---
DEFAULT_MAX_STEPS = 500          # Default maximum steps per episode
DEFAULT_PASSENGER_RATE = 0.1     # Default probability of a new passenger appearing per step
DEFAULT_NUM_ELEVATORS = 3        # Default number of elevators in the simulation
DEFAULT_NUM_FLOORS = 5           # Default number of floors in the building
INITIAL_PASSENGERS = 5           # Number of passengers to generate at the start of an episode
RENDER_FPS = 60                  # Frames per second for rendering the simulation

# --- Rendering Dimensions ---
SCREEN_WIDTH = 800               # Width of the simulation window in pixels
SCREEN_HEIGHT = 600              # Height of the simulation window in pixels
FLOOR_HEIGHT = 60                # Height of each floor in pixels
ELEVATOR_WIDTH = 40              # Width of each elevator in pixels
ELEVATOR_SPACING = 20            # Spacing between elevators in pixels
Y_OFFSET = 50                    # Vertical offset from the bottom of the screen for rendering

# --- Reward Component Weights ---
# These weights determine the importance of each reward component in the total reward calculation.
WEIGHT_MOVEMENT = 0.3            # Importance of efficient elevator movement
WEIGHT_PICKUP = 0.2              # Importance of timely passenger pickups
WEIGHT_DELIVERY = 0.15           # Importance of quick passenger deliveries
WEIGHT_BUTTON_RESPONSE = 0.1     # Importance of responding to floor calls

# --- Movement-Related Rewards and Penalties ---
IDLE_WITH_PASSENGERS_PENALTY = -1.0  # Penalty for an elevator being idle while carrying passengers
MOVING_TO_DESTINATION_BONUS = 0.5    # Reward for moving towards a passenger's destination
MOVING_WRONG_DIRECTION_PENALTY = -0.8 # Penalty for moving away from a passenger's destination
MOVING_WITHOUT_PASSENGERS_PENALTY = -0.5 # Penalty for moving without a destination or pickup target
MOVING_TO_PICKUP_BONUS = 0.3         # Reward for moving towards a floor with a waiting passenger
MOVING_TO_PICKUP_SMALL_BONUS = 0.1   # Smaller reward if another elevator is already handling the call

# --- Pickup and Delivery Rewards ---
PICKUP_BASE_REWARD = 2.0             # Base reward for picking up a passenger
PICKUP_WAIT_MULTIPLIER = 0.2         # Multiplier that increases pickup reward based on wait time
MAX_PICKUP_MULTIPLIER = 5.0          # Maximum multiplier for the pickup reward

DELIVERY_BASE_REWARD = 7.5           # Base reward for successfully delivering a passenger
DELIVERY_BONUS_SPEED = 5.0           # Bonus for fast delivery
DELIVERY_BONUS_WAIT = 2.5            # Bonus for minimizing passenger wait time before pickup
DELIVERY_BONUS_EFFICIENCY = 0.25     # Bonus for efficient travel (beating expected time)

# --- Button Response Rewards ---
BUTTON_RESPONSE_BASE = 0.1           # Base reward for moving towards a floor with a pressed call button
BUTTON_RESPONSE_WAIT_MULTIPLIER = 0.1 # Multiplier that increases reward based on how long the button has been active
MAX_BUTTON_RESPONSE_MULTIPLIER = 5.0 # Maximum multiplier for the button response reward
LONG_WAIT_THRESHOLD = 5              # Number of steps after which a waiting passenger is considered a long wait

# --- Time-Based Penalties ---
TIME_PENALTY_BASE = -0.15            # Base penalty applied for each step a passenger is waiting
TIME_PENALTY_MULTIPLIER = 0.2        # Multiplier that increases the penalty over time
MAX_TIME_PENALTY = -5.0              # Maximum penalty per passenger to avoid excessive negative rewards

# --- Elevator State Enums ---
ELEVATOR_IDLE = 0
ELEVATOR_UP = 1
ELEVATOR_DOWN = 2

# --- Color Definitions for Rendering ---
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'GRAY': (128, 128, 128),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'CYAN': (0, 255, 255),
    'MAGENTA': (255, 0, 255)
} 