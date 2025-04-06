"""
Multi-Elevator Environment

A Gymnasium environment simulating multiple elevators serving multiple floors.
The goal is to minimize passenger waiting time through efficient elevator control.
"""
from typing import Dict, Any, Optional, Tuple
import logging

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .constants import *

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MultiElevatorEnv(gym.Env):
    """
    Multi-Elevator Environment

    A Gymnasium environment that simulates multiple elevators serving multiple floors.
    The environment supports configurable number of elevators and floors, with the goal
    of minimizing passenger waiting time through efficient elevator control.

    Attributes:
        num_floors (int): Number of floors in the building
        num_elevators (int): Number of elevators available
        max_steps (int): Maximum number of steps before episode termination
        passenger_rate (float): Probability of new passenger arrival per step
        observation_space (spaces.Dict): The observation space definition
        action_space (spaces.MultiDiscrete): The action space definition
        render_mode (Optional[str]): The rendering mode ('human' or 'rgb_array')
        current_reward (float): The most recent reward received

    State Variables:
        elevator_positions (np.ndarray): Current floor position of each elevator
        elevator_directions (np.ndarray): Current direction of each elevator (0=idle, 1=up, 2=down)
        elevator_buttons (np.ndarray): Button states inside each elevator
        floor_buttons (np.ndarray): Call button states on each floor
        waiting_passengers (np.ndarray): Number of waiting passengers on each floor
        passengers_in_elevators (List[List[str]]): Passengers currently in each elevator
        waiting_time (Dict[str, Dict[str, Any]]): Tracking data for passenger wait times
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = DEFAULT_MAX_STEPS,
                 passenger_rate: float = DEFAULT_PASSENGER_RATE, num_elevators: int = DEFAULT_NUM_ELEVATORS, 
                 num_floors: int = DEFAULT_NUM_FLOORS) -> None:
        """
        Initialize the Multi-Elevator environment.

        Args:
            render_mode: Optional rendering mode ('human' or 'rgb_array')
            max_steps: Maximum number of steps before episode termination
            passenger_rate: Probability of new passenger arrival per step
            num_elevators: Number of elevators in the building
            num_floors: Number of floors in the building
        """
        super().__init__()
        
        # Environment configuration
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.max_steps = max_steps
        self.passenger_rate = passenger_rate
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "elevator_positions": spaces.MultiDiscrete([self.num_floors] * self.num_elevators),
            "elevator_directions": spaces.MultiDiscrete([3] * self.num_elevators),  # 0: idle, 1: up, 2: down
            "elevator_buttons": spaces.MultiBinary([self.num_floors, self.num_elevators]),
            "floor_buttons": spaces.MultiBinary(self.num_floors),
            "waiting_passengers": spaces.Box(low=0, high=100, shape=(self.num_floors,), dtype=np.int32)
        })
        
        # Define action space: For each elevator: 0 (idle), 1 (up), 2 (down)
        self.action_space = spaces.MultiDiscrete([3] * self.num_elevators)
        
        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        # Initialize state variables
        self.current_reward = 0.0
        self.elevator_positions = None
        self.elevator_directions = None
        self.elevator_buttons = None
        self.floor_buttons = None
        self.waiting_passengers = None
        self.passengers_in_elevators = None
        self.waiting_time = None
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        
        # Reset to initialize state variables
        self.reset()
        
    def _get_obs(self) -> Dict[str, Any]:
        """Return the current observation of the environment."""
        return {
            "elevator_positions": self.elevator_positions.copy(),
            "elevator_directions": self.elevator_directions.copy(),
            "elevator_buttons": self.elevator_buttons.copy(),
            "floor_buttons": self.floor_buttons.copy(),
            "waiting_passengers": self.waiting_passengers.copy()
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional information about the current state."""
        passengers_in_elevators_count = sum(len(passengers) for passengers in self.passengers_in_elevators)
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)
        
        return {
            "waiting_passengers": int(np.sum(self.waiting_passengers)),
            "passengers_in_elevators": passengers_in_elevators_count,
            "average_waiting_time": avg_wait_time,
            "total_passengers_served": self.total_passengers_served,
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility
            options: Optional configuration overrides

        Returns:
            tuple: (observation, info)
                observation: Current state observation
                info: Additional information about the environment
        """
        super().reset(seed=seed)
        logger.debug("Resetting MultiElevatorEnv")
        
        # Initialize state arrays
        self.elevator_positions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_directions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_buttons = np.zeros((self.num_floors, self.num_elevators), dtype=bool)
        self.floor_buttons = np.zeros(self.num_floors, dtype=bool)
        self.waiting_passengers = np.zeros(self.num_floors, dtype=np.int32)
        
        # Initialize tracking variables
        self.passengers_in_elevators = [[] for _ in range(self.num_elevators)]
        self.waiting_time = {}
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        self.current_reward = 0.0
        
        # Distribute elevators and generate initial passengers
        self._distribute_elevators()
        for _ in range(INITIAL_PASSENGERS):
            self._generate_passenger()
        
        # Initialize rendering if needed
        if self.render_mode == "human" and self.screen is None:
            self._render_setup()
            
        return self._get_obs(), self._get_info()
    
    def _distribute_elevators(self) -> None:
        """Distribute elevators evenly across floors."""
        if self.num_elevators > 1:
            self.elevator_positions[0] = 0
            self.elevator_positions[-1] = self.num_floors - 1
            for i in range(1, self.num_elevators - 1):
                self.elevator_positions[i] = (i * self.num_floors) // self.num_elevators
    
    def _generate_passenger(self, start_floor: Optional[int] = None) -> str:
        """Generate a new passenger with random start and destination floors."""
        if start_floor is None:
            start_floor = self.np_random.integers(0, self.num_floors)
        elif start_floor < 0 or start_floor >= self.num_floors:
            raise ValueError(f"Invalid start floor: {start_floor}")
            
        destination_floor = start_floor
        while destination_floor == start_floor:
            destination_floor = self.np_random.integers(0, self.num_floors)
        
        self.waiting_passengers[start_floor] += 1
        self.floor_buttons[start_floor] = True
        
        passenger_id = f"{self.current_step}_{start_floor}_{destination_floor}_{self.waiting_passengers[start_floor]}"
        self.waiting_time[passenger_id] = {
            "start_floor": start_floor,
            "destination_floor": destination_floor,
            "wait_start": self.current_step,
            "wait_end": None,
            "travel_end": None,
            "elevator_id": None
        }
        logger.debug("Generated passenger %s at floor %d with destination %d", passenger_id, start_floor, destination_floor)
        return passenger_id
    
    def calculate_reward(self) -> float:
        """
        Calculate the total reward for the current step.
        
        The reward is a weighted sum of:
        - Movement rewards (efficient elevator movement)
        - Pickup rewards (timely passenger pickup)
        - Delivery rewards (efficient passenger delivery)
        - Button response rewards (responding to call buttons)
        - Time penalties (applied directly, not weighted)
        
        Returns:
            float: The total reward for this step
        """
        logger.debug("--- Calculating reward at step %d ---", self.current_step)

        # Calculate individual reward components
        movement_reward = self._calculate_movement_reward()
        pickup_reward = self._calculate_pickup_reward()
        delivery_reward = self._calculate_delivery_reward()
        time_penalty = self._calculate_time_penalty()
        button_reward = self._calculate_button_response_reward()

        # Log individual components
        logger.debug(
            f"Raw Rewards: Move={movement_reward:.2f}, Pickup={pickup_reward:.2f}, "
            f"Deliver={delivery_reward:.2f}, Button={button_reward:.2f}, "
            f"TimePen={time_penalty:.2f}"
        )

        # Calculate weighted total
        total_reward = (
            WEIGHT_MOVEMENT * movement_reward +
            WEIGHT_PICKUP * pickup_reward +
            WEIGHT_DELIVERY * delivery_reward +
            WEIGHT_BUTTON_RESPONSE * button_reward +
            time_penalty
        )

        # Log weighted components
        logger.debug(
            f"Weighted Total Reward: {total_reward:.2f} "
            f"(W_Move={WEIGHT_MOVEMENT*movement_reward:.2f}, "
            f"W_Pickup={WEIGHT_PICKUP*pickup_reward:.2f}, "
            f"W_Deliver={WEIGHT_DELIVERY*delivery_reward:.2f}, "
            f"W_Button={WEIGHT_BUTTON_RESPONSE*button_reward:.2f}, "
            f"TimePen={time_penalty:.2f})"
        )
        return total_reward

    def _calculate_movement_reward(self) -> float:
        """
        Calculate reward based on elevator movement patterns.
        
        Rewards:
        - Moving towards passenger destinations (+)
        - Moving towards waiting passengers (+)
        - Moving in wrong direction (-)
        - Being idle with passengers (-)
        - Moving without purpose (-)
        
        Returns:
            float: The movement reward component
        """
        movement_reward = 0.0
        
        for elev_id in range(self.num_elevators):
            elev_pos = self.elevator_positions[elev_id]
            elev_dir = self.elevator_directions[elev_id]
            has_passengers = len(self.passengers_in_elevators[elev_id]) > 0
            
            # Handle elevators with passengers
            if has_passengers:
                if elev_dir == ELEVATOR_IDLE:
                    movement_reward += IDLE_WITH_PASSENGERS_PENALTY
                elif self._is_moving_to_destination(elev_id):
                    movement_reward += MOVING_TO_DESTINATION_BONUS
                else:
                    movement_reward += self._calculate_direction_penalty(elev_id, elev_pos, elev_dir)
            # Handle empty elevators
            elif elev_dir != ELEVATOR_IDLE:
                movement_reward += self._calculate_empty_movement_reward(elev_id, elev_pos, elev_dir)

        if movement_reward != 0.0:
            logger.debug(f"Movement reward: {movement_reward:.2f}")
        return movement_reward

    def _calculate_direction_penalty(self, elev_id: int, elev_pos: int, elev_dir: int) -> float:
        """Calculate penalty for moving in wrong direction with passengers."""
        if elev_dir == ELEVATOR_UP:
            has_dest_above = any(
                self.waiting_time[p_id]["destination_floor"] > elev_pos
                for p_id in self.passengers_in_elevators[elev_id]
                if p_id in self.waiting_time
            )
            if not has_dest_above and not any(self.waiting_passengers[floor] > 0 
                                            for floor in range(elev_pos + 1, self.num_floors)):
                return MOVING_WRONG_DIRECTION_PENALTY
        elif elev_dir == ELEVATOR_DOWN:
            has_dest_below = any(
                self.waiting_time[p_id]["destination_floor"] < elev_pos
                for p_id in self.passengers_in_elevators[elev_id]
                if p_id in self.waiting_time
            )
            if not has_dest_below and not any(self.waiting_passengers[floor] > 0 
                                           for floor in range(0, elev_pos)):
                return MOVING_WRONG_DIRECTION_PENALTY
        return 0.0

    def _calculate_empty_movement_reward(self, elev_id: int, elev_pos: int, elev_dir: int) -> float:
        """Calculate reward for empty elevator movement."""
        if elev_dir == ELEVATOR_UP:
            waiting_floors_above = [floor for floor in range(elev_pos + 1, self.num_floors) 
                                  if self.waiting_passengers[floor] > 0]
            if waiting_floors_above:
                closest_floor = min(waiting_floors_above)
                other_elevators_heading_up = [
                    i for i in range(self.num_elevators)
                    if i != elev_id and self.elevator_directions[i] == ELEVATOR_UP 
                    and self.elevator_positions[i] < closest_floor
                ]
                if not other_elevators_heading_up or all(self.elevator_positions[i] <= elev_pos 
                                                       for i in other_elevators_heading_up):
                    return MOVING_TO_PICKUP_BONUS
                return MOVING_TO_PICKUP_SMALL_BONUS
            return MOVING_WITHOUT_PASSENGERS_PENALTY
            
        elif elev_dir == ELEVATOR_DOWN:
            waiting_floors_below = [floor for floor in range(0, elev_pos) 
                                  if self.waiting_passengers[floor] > 0]
            if waiting_floors_below:
                closest_floor = max(waiting_floors_below)
                other_elevators_heading_down = [
                    i for i in range(self.num_elevators)
                    if i != elev_id and self.elevator_directions[i] == ELEVATOR_DOWN 
                    and self.elevator_positions[i] > closest_floor
                ]
                if not other_elevators_heading_down or all(self.elevator_positions[i] >= elev_pos 
                                                         for i in other_elevators_heading_down):
                    return MOVING_TO_PICKUP_BONUS
                return MOVING_TO_PICKUP_SMALL_BONUS
            return MOVING_WITHOUT_PASSENGERS_PENALTY
            
        return 0.0

    def _calculate_pickup_reward(self) -> float:
        """
        Calculate reward for picking up passengers.
        
        The reward increases with waiting time but is capped to prevent
        exploitation of extremely long waits.
        
        Returns:
            float: The pickup reward component
        """
        pickup_reward = 0.0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] == self.current_step:
                waiting_duration = data["wait_end"] - data["wait_start"]
                pickup_reward += PICKUP_BASE_REWARD * min(
                    MAX_PICKUP_MULTIPLIER,
                    1.0 + PICKUP_WAIT_MULTIPLIER * waiting_duration
                )

        if pickup_reward != 0.0:
            logger.debug(f"Pickup reward: {pickup_reward:.2f}")
        return pickup_reward

    def _calculate_delivery_reward(self) -> float:
        """
        Calculate reward for delivering passengers to their destinations.

        The reward includes:
        - Base reward for successful delivery
        - Time efficiency bonus
        - Waiting time penalty
        
        Returns:
            float: The delivery reward component
        """
        delivery_reward = 0.0
        for passenger_id, data in self.waiting_time.items():
            # Check if passenger exists and was delivered this step
            if data["travel_end"] == self.current_step and passenger_id in self.waiting_time:
                # Ensure passenger data is valid before proceeding
                if data["wait_end"] is None or data["wait_start"] is None or \
                   data["destination_floor"] is None or data["start_floor"] is None:
                    logger.warning(f"Incomplete data for delivered passenger {passenger_id}. Skipping reward calc.")
                    continue  # Skip this passenger if data is incomplete

                # Base reward for successful delivery
                delivery_reward += DELIVERY_BASE_REWARD

                # Calculate time-based components
                wait_time = data["wait_end"] - data["wait_start"]
                travel_time = data["travel_end"] - data["wait_end"]

                # Handle potential edge case where travel_time might be zero or negative
                if travel_time <= 0:
                    travel_time = 1  # Assign a minimum travel time if happens in same step

                floor_distance = abs(data["destination_floor"] - data["start_floor"])
                # Ensure minimum expected time even for same floor
                expected_travel_time = floor_distance + 1

                # Calculate efficiency bonus
                time_efficiency = max(0, expected_travel_time - travel_time)
                time_bonus = (
                    DELIVERY_BONUS_SPEED * (0.9 ** max(0, travel_time - expected_travel_time)) +
                    DELIVERY_BONUS_WAIT * (0.9 ** wait_time) +
                    DELIVERY_BONUS_EFFICIENCY * time_efficiency
                )
                delivery_reward += time_bonus

        if delivery_reward != 0.0:
            logger.debug(f"Delivery reward: {delivery_reward:.2f}")
        return delivery_reward

    def _calculate_time_penalty(self) -> float:
        """
        Calculate penalty for passengers waiting too long.
        
        The penalty increases with waiting time but is capped to prevent
        extreme negative rewards.
        
        Returns:
            float: The time penalty component (negative value)
        """
        time_penalty = 0.0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] is None:
                waiting_duration = self.current_step - data["wait_start"]
                penalty = TIME_PENALTY_BASE * (1.0 + TIME_PENALTY_MULTIPLIER * waiting_duration)
                time_penalty += max(MAX_TIME_PENALTY, penalty)

        if time_penalty != 0.0:
            logger.debug(f"Time penalty: {time_penalty:.2f}")
        return time_penalty

    def _calculate_button_response_reward(self) -> float:
        """
        Calculate reward for responding to floor button calls.
        
        The reward increases for:
        - Moving towards floors with long-waiting passengers
        - Efficient distribution of elevators to calls
        
        Returns:
            float: The button response reward component
        """
        button_reward = 0.0
        for floor in range(self.num_floors):
            if self.floor_buttons[floor]:
                longest_wait = max(
                    (self.current_step - data["wait_start"]
                     for passenger_id, data in self.waiting_time.items()
                     if data["start_floor"] == floor and data["wait_end"] is None),
                    default=0
                )
                if longest_wait > LONG_WAIT_THRESHOLD:
                    for elev_id in range(self.num_elevators):
                        elev_pos = self.elevator_positions[elev_id]
                        elev_dir = self.elevator_directions[elev_id]
                        if ((elev_dir == ELEVATOR_UP and floor > elev_pos) or 
                            (elev_dir == ELEVATOR_DOWN and floor < elev_pos)):
                            button_reward += BUTTON_RESPONSE_BASE * min(
                                MAX_BUTTON_RESPONSE_MULTIPLIER,
                                1.0 + BUTTON_RESPONSE_WAIT_MULTIPLIER * longest_wait
                            )

        if button_reward != 0.0:
            logger.debug(f"Button response reward: {button_reward:.2f}")
        return button_reward

    def _is_moving_to_destination(self, elevator_id: int) -> bool:
        curr_pos = self.elevator_positions[elevator_id]
        curr_dir = self.elevator_directions[elevator_id]
        if curr_dir == 0:
            return False
        for passenger_id in self.passengers_in_elevators[elevator_id]:
            if passenger_id not in self.waiting_time:
                continue
            dest_floor = self.waiting_time[passenger_id]["destination_floor"]
            if curr_dir == 1 and dest_floor > curr_pos:
                return True
            elif curr_dir == 2 and dest_floor < curr_pos:
                return True
        return False

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Array of elevator actions (0=idle, 1=up, 2=down)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                observation: Current state observation
                reward: Reward for this step
                terminated: Whether episode is done
                truncated: Whether episode was artificially terminated
                info: Additional information
        """
        self.current_step += 1
        
        # Process elevator actions
        for elev_id in range(self.num_elevators):
            self._process_elevator_action(elev_id, action[elev_id])
            
        # Generate new passengers
        if self.np_random.random() < self.passenger_rate:
            self._generate_passenger()
            
        # Calculate reward and update state
        reward = self.calculate_reward()
        self.current_reward = reward
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _process_elevator_action(self, elevator_id: int, action: int) -> None:
        """
        Process an elevator's action and update its state.
        
        Args:
            elevator_id: ID of the elevator to process
            action: Action to take (0=idle, 1=up, 2=down)
        """
        elev_pos = self.elevator_positions[elevator_id]
        
        # Update position and direction based on action
        if action == ELEVATOR_UP and elev_pos < self.num_floors - 1:
            self.elevator_positions[elevator_id] += 1
            self.elevator_directions[elevator_id] = ELEVATOR_UP
        elif action == ELEVATOR_DOWN and elev_pos > 0:
            self.elevator_positions[elevator_id] -= 1
            self.elevator_directions[elevator_id] = ELEVATOR_DOWN
        else:
            self.elevator_directions[elevator_id] = ELEVATOR_IDLE
            
        # Process passenger interactions at current floor
        self._process_floor(elevator_id)
    
    def _process_floor(self, elevator_id: int) -> None:
        """
        Process passenger interactions at an elevator's current floor.
        
        Handles both passenger dropoff and pickup operations.
        
        Args:
            elevator_id: ID of the elevator to process
        """
        current_floor = self.elevator_positions[elevator_id]
        self._process_dropoff(elevator_id, current_floor)
        self._process_pickup(elevator_id, current_floor)
    
    def _process_dropoff(self, elevator_id: int, current_floor: int) -> None:
        """
        Process passenger dropoff at the current floor.
        
        Updates passenger tracking data and elevator button states.
        
        Args:
            elevator_id: ID of the elevator
            current_floor: Current floor number
        """
        # Find passengers to drop off
        passengers_to_remove = [
            passenger_id for passenger_id in self.passengers_in_elevators[elevator_id]
            if self.waiting_time[passenger_id]["destination_floor"] == current_floor
        ]
        
        # Process each passenger
        for passenger_id in passengers_to_remove:
            # Remove from elevator
            self.passengers_in_elevators[elevator_id].remove(passenger_id)
            
            # Update tracking data
            self.waiting_time[passenger_id]["travel_end"] = self.current_step
            self.total_waiting_time += (
                self.waiting_time[passenger_id]["travel_end"] -
                self.waiting_time[passenger_id]["wait_start"]
            )
            self.total_passengers_served += 1
        
        # Update elevator button state
        if not any(
            self.waiting_time[p]["destination_floor"] == current_floor
            for p in self.passengers_in_elevators[elevator_id]
        ):
            self.elevator_buttons[current_floor, elevator_id] = False
    
    def _process_pickup(self, elevator_id: int, current_floor: int) -> None:
        """
        Process passenger pickup at the current floor.
        
        Updates passenger tracking data, elevator and floor button states.
        
        Args:
            elevator_id: ID of the elevator
            current_floor: Current floor number
        """
        if self.waiting_passengers[current_floor] > 0:
            # Find waiting passengers
            passengers_to_pickup = [
                passenger_id for passenger_id, data in self.waiting_time.items()
                if (data["start_floor"] == current_floor and 
                    data["wait_end"] is None and 
                    data["elevator_id"] is None)
            ]
            
            # Process each passenger
            for passenger_id in passengers_to_pickup:
                # Add to elevator
                self.passengers_in_elevators[elevator_id].append(passenger_id)
                
                # Update tracking data
                self.waiting_time[passenger_id]["wait_end"] = self.current_step
                self.waiting_time[passenger_id]["elevator_id"] = elevator_id
                
                # Update button states
                dest_floor = self.waiting_time[passenger_id]["destination_floor"]
                self.elevator_buttons[dest_floor, elevator_id] = True
                self.waiting_passengers[current_floor] -= 1
            
            # Update floor button state
            self.floor_buttons[current_floor] = self.waiting_passengers[current_floor] > 0
    
    def _render_setup(self) -> None:
        """Initialize the Pygame rendering environment."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-Elevator Environment")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
    
    def _render_frame(self) -> None:
        """
        Render the current frame of the environment.
        
        Draws:
        - Floor lines and labels
        - Floor call buttons
        - Waiting passenger counts
        - Elevators with passenger counts
        - Direction indicators
        - Status information
        """
        self.screen.fill(COLORS['WHITE'])
        
        # Calculate dimensions
        floor_height = FLOOR_HEIGHT
        y_offset = Y_OFFSET
        
        # Draw environment components
        self._draw_floors(floor_height, y_offset)
        self._draw_elevators(ELEVATOR_SPACING, ELEVATOR_WIDTH, floor_height, floor_height, y_offset)
        self._draw_info()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _draw_floors(self, floor_height: int, y_offset: int) -> None:
        """
        Draw floor lines, labels, call buttons, and waiting passenger counts.
        
        Args:
            floor_height: Height of each floor in pixels
            y_offset: Vertical offset from bottom of screen
        """
        for floor in range(self.num_floors):
            # Calculate floor position
            y = self.screen_height - (floor + 1) * floor_height - y_offset
            
            # Draw floor line
            pygame.draw.line(
                self.screen, 
                COLORS['GRAY'],
                (0, y + floor_height), 
                (self.screen_width, y + floor_height), 
                2
            )
            
            # Draw floor label
            text = self.font.render(f"Floor {floor}", True, COLORS['BLACK'])
            text_rect = text.get_rect(midleft=(10, y + floor_height // 2))
            self.screen.blit(text, text_rect)
            
            # Draw call button
            button_color = COLORS['GREEN'] if self.floor_buttons[floor] else COLORS['RED']
            pygame.draw.circle(
                self.screen, 
                button_color,
                (self.screen_width - 30, y + floor_height // 2), 
                12
            )
            
            # Draw waiting passenger count if any
            if self.waiting_passengers[floor] > 0:
                text = self.font.render(
                    f"Waiting: {self.waiting_passengers[floor]}", 
                    True, 
                    COLORS['BLACK']
                )
                text_rect = text.get_rect(midright=(self.screen_width - 50, y + floor_height // 2))
                self.screen.blit(text, text_rect)
    
    def _draw_elevators(self, elevator_spacing: int, elevator_width: int, 
                       elevator_height: int, floor_height: int, y_offset: int) -> None:
        """
        Draw elevators with their current state.
        
        Args:
            elevator_spacing: Horizontal spacing between elevators
            elevator_width: Width of each elevator
            elevator_height: Height of each elevator
            floor_height: Height of each floor
            y_offset: Vertical offset from bottom of screen
        """
        for elev_id in range(self.num_elevators):
            # Calculate elevator position
            elevator_x = (elev_id + 1) * elevator_spacing - elevator_width // 2
            elevator_y = self.screen_height - (self.elevator_positions[elev_id] + 1) * floor_height - y_offset
            
            # Draw elevator box
            elevator_color = COLORS['BLUE']  # Could add different colors for different states
            pygame.draw.rect(
                self.screen, 
                elevator_color,
                (elevator_x, elevator_y, elevator_width, elevator_height)
            )
            
            # Draw elevator ID and passenger count
            text = self.font.render(
                f"E{elev_id} (P:{len(self.passengers_in_elevators[elev_id])})", 
                True, 
                COLORS['WHITE']
            )
            text_rect = text.get_rect(center=(elevator_x + elevator_width // 2, 
                                            elevator_y + elevator_height // 2))
            self.screen.blit(text, text_rect)
            
            # Draw direction indicator
            direction_symbol = "↑" if self.elevator_directions[elev_id] == ELEVATOR_UP else \
                             "↓" if self.elevator_directions[elev_id] == ELEVATOR_DOWN else "•"
            dir_text = self.font.render(direction_symbol, True, COLORS['WHITE'])
            dir_rect = dir_text.get_rect(center=(elevator_x + elevator_width // 2, 
                                               elevator_y + elevator_height - 15))
            self.screen.blit(dir_text, dir_rect)
    
    def _draw_info(self) -> None:
        """Draw status information at the top of the screen."""
        # Calculate average waiting time
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)
        
        # Prepare info texts
        info_text = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Total served: {self.total_passengers_served}",
            f"Avg wait time: {avg_wait_time:.2f} steps",
            f"Reward: {self.current_reward:.2f}"
        ]
        
        # Draw each info text
        for i, text in enumerate(info_text):
            surf = self.font.render(text, True, COLORS['BLACK'])
            rect = surf.get_rect(topleft=(10 + i * (self.screen_width // len(info_text)), 10))
            self.screen.blit(surf, rect)
    
    def close(self) -> None:
        """Clean up the Pygame environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
