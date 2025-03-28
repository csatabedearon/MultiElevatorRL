"""
Constants and configuration values for the MultiElevator environment.
"""

# Environment Configuration
DEFAULT_MAX_STEPS = 500
DEFAULT_PASSENGER_RATE = 0.1
DEFAULT_NUM_ELEVATORS = 3
DEFAULT_NUM_FLOORS = 5
INITIAL_PASSENGERS = 5
RENDER_FPS = 60  # Changed from 1000 to a more reasonable value

# Screen Dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FLOOR_HEIGHT = 60
ELEVATOR_WIDTH = 40
ELEVATOR_SPACING = 20
Y_OFFSET = 50

# Reward Weights
WEIGHT_MOVEMENT = 0.3
WEIGHT_PICKUP = 0.2
WEIGHT_DELIVERY = 0.3
WEIGHT_BUTTON_RESPONSE = 0.1

# Movement Rewards
IDLE_WITH_PASSENGERS_PENALTY = -1.0
MOVING_TO_DESTINATION_BONUS = 0.5
MOVING_WRONG_DIRECTION_PENALTY = -0.8
MOVING_WITHOUT_PASSENGERS_PENALTY = -0.5
MOVING_TO_PICKUP_BONUS = 0.3
MOVING_TO_PICKUP_SMALL_BONUS = 0.1

# Pickup/Delivery Rewards
PICKUP_BASE_REWARD = 2.0
PICKUP_WAIT_MULTIPLIER = 0.2
MAX_PICKUP_MULTIPLIER = 5.0

# Button Response Rewards
BUTTON_RESPONSE_BASE = 0.1
BUTTON_RESPONSE_WAIT_MULTIPLIER = 0.1
MAX_BUTTON_RESPONSE_MULTIPLIER = 5.0
LONG_WAIT_THRESHOLD = 5

# Time Penalties
TIME_PENALTY_BASE = -0.1
TIME_PENALTY_MULTIPLIER = 0.2
MAX_TIME_PENALTY = -5.0

# Elevator States
ELEVATOR_IDLE = 0
ELEVATOR_UP = 1
ELEVATOR_DOWN = 2

# Colors
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