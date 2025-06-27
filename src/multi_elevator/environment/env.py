"""
This module defines the core Multi-Elevator Reinforcement Learning environment.

The `MultiElevatorEnv` class, compliant with the Gymnasium API, simulates a building
with multiple elevators and floors. The agent's goal is to learn an efficient
policy for controlling the elevators to minimize passenger wait times.

The environment features a configurable number of elevators and floors, a detailed
reward system, and support for both random passenger generation and predefined
scenarios. It also includes a Pygame-based rendering mode for visualization.
"""
from typing import Dict, Any, Optional, Tuple
import logging

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .constants import *

# Configure logging for the environment module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MultiElevatorEnv(gym.Env):
    """
    A Gymnasium environment for simulating a multi-elevator system.

    This environment models a building with a specified number of floors and elevators.
    The agent's task is to control the movement of each elevator (up, down, or idle)
    at each time step to serve passengers efficiently.

    The observation space includes the state of elevators (positions, directions),
    floor call buttons, and buttons pressed inside elevators. The action space is a
    discrete set of actions for each elevator.

    The reward function is designed to incentivize minimizing passenger wait times,
    encouraging timely pickups, fast deliveries, and intelligent routing.

    Attributes:
        num_floors (int): The number of floors in the building.
        num_elevators (int): The number of elevators in the system.
        max_steps (int): The maximum number of steps per episode.
        passenger_rate (float): The probability of a new passenger arriving per step.
        render_mode (Optional[str]): The mode for rendering ('human' or 'rgb_array').
        observation_space (spaces.Dict): The definition of the observation space.
        action_space (spaces.MultiDiscrete): The definition of the action space.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = DEFAULT_MAX_STEPS,
                 passenger_rate: float = DEFAULT_PASSENGER_RATE, num_elevators: int = DEFAULT_NUM_ELEVATORS,
                 num_floors: int = DEFAULT_NUM_FLOORS, scenario: Optional[list] = None) -> None:
        """
        Initializes the Multi-Elevator environment.

        Args:
            render_mode: The mode for rendering ('human' or 'rgb_array').
            max_steps: The maximum number of steps per episode.
            passenger_rate: The probability of a new passenger arriving at a random floor.
                          This is ignored if a scenario is provided.
            num_elevators: The number of elevators in the simulation.
            num_floors: The number of floors in the building.
            scenario: An optional list of dictionaries defining a predefined passenger
                      arrival sequence. Each dictionary should have 'arrival_step',
                      'start_floor', and 'destination_floor'.
        """
        super().__init__()

        # --- Environment Configuration ---
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.max_steps = max_steps
        self.passenger_rate = passenger_rate

        # --- Scenario Handling ---
        # If a scenario is provided, sort it by arrival time for efficient processing.
        if scenario:
            self.scenario = sorted(scenario, key=lambda p: p['arrival_step'])
        else:
            self.scenario = None
        self.scenario_index = 0

        # --- Observation and Action Spaces ---
        # The observation space captures all relevant information about the environment's state.
        self.observation_space = spaces.Dict({
            # Position of each elevator (floor number)
            "elevator_positions": spaces.MultiDiscrete([self.num_floors] * self.num_elevators),
            # Direction of each elevator (0: idle, 1: up, 2: down)
            "elevator_directions": spaces.MultiDiscrete([3] * self.num_elevators),
            # Buttons pressed inside each elevator for destination floors
            "elevator_buttons": spaces.MultiBinary([self.num_floors, self.num_elevators]),
            # Call buttons pressed on each floor
            "floor_buttons": spaces.MultiBinary(self.num_floors),
            # Number of passengers waiting on each floor
            "waiting_passengers": spaces.Box(low=0, high=100, shape=(self.num_floors,), dtype=np.int32)
        })

        # The action space defines the possible actions for each elevator.
        # For each elevator, the agent can choose: 0 (idle), 1 (move up), 2 (move down).
        self.action_space = spaces.MultiDiscrete([3] * self.num_elevators)

        # --- Rendering Setup ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None

        # --- State Variables ---
        # These are initialized here and reset in the `reset` method.
        self.current_reward = 0.0
        self.elevator_positions: Optional[np.ndarray] = None
        self.elevator_directions: Optional[np.ndarray] = None
        self.elevator_buttons: Optional[np.ndarray] = None
        self.floor_buttons: Optional[np.ndarray] = None
        self.waiting_passengers: Optional[np.ndarray] = None
        self.passengers_in_elevators: Optional[list] = None
        self.waiting_time: Optional[dict] = None
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0

        # Initialize state variables by calling reset
        self.reset()
        
    def _get_obs(self) -> Dict[str, Any]:
        """Constructs the observation dictionary from the current environment state."""
        return {
            "elevator_positions": self.elevator_positions.copy(),
            "elevator_directions": self.elevator_directions.copy(),
            "elevator_buttons": self.elevator_buttons.copy(),
            "floor_buttons": self.floor_buttons.copy(),
            "waiting_passengers": self.waiting_passengers.copy()
        }

    def _get_info(self) -> Dict[str, Any]:
        """Provides auxiliary information about the environment's state."""
        passengers_in_elevators_count = sum(len(p) for p in self.passengers_in_elevators)
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)

        return {
            "waiting_passengers": int(np.sum(self.waiting_passengers)),
            "passengers_in_elevators": passengers_in_elevators_count,
            "average_waiting_time": avg_wait_time,
            "total_passengers_served": self.total_passengers_served,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Resets the environment to its initial state.

        This method is called at the beginning of each new episode. It reinitializes
        all state variables, including elevator positions, passenger data, and step counters.

        Args:
            seed: An optional random seed for reproducibility.
            options: Optional dictionary for additional configuration (not used).

        Returns:
            A tuple containing the initial observation and auxiliary information.
        """
        super().reset(seed=seed)
        logger.debug("Resetting environment to initial state.")

        # Initialize state arrays for elevators and floors
        self.elevator_positions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_directions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_buttons = np.zeros((self.num_floors, self.num_elevators), dtype=bool)
        self.floor_buttons = np.zeros(self.num_floors, dtype=bool)
        self.waiting_passengers = np.zeros(self.num_floors, dtype=np.int32)

        # Reset tracking variables for the new episode
        self.passengers_in_elevators = [[] for _ in range(self.num_elevators)]
        self.waiting_time = {}
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        self.current_reward = 0.0
        self.scenario_index = 0  # Reset progress for scenario-based episodes

        # Distribute elevators to their starting positions
        self._distribute_elevators()

        # Generate initial passengers if not running a predefined scenario
        if not self.scenario:
            for _ in range(INITIAL_PASSENGERS):
                self._generate_passenger()

        # Set up the rendering window if in human mode
        if self.render_mode == "human" and self.screen is None:
            self._render_setup()

        return self._get_obs(), self._get_info()
    
    def _distribute_elevators(self) -> None:
        """Distributes elevators evenly across the floors at the start of an episode."""
        if self.num_elevators > 1:
            # Place the first elevator at the bottom floor and the last at the top floor
            self.elevator_positions[0] = 0
            self.elevator_positions[-1] = self.num_floors - 1
            # Spread the remaining elevators evenly in between
            for i in range(1, self.num_elevators - 1):
                self.elevator_positions[i] = (i * self.num_floors) // self.num_elevators

    def _generate_passenger(self, start_floor: Optional[int] = None, destination_floor: Optional[int] = None) -> str:
        """
        Generates a new passenger and adds them to the waiting list.

        If start and destination floors are not provided, they are chosen randomly.
        This method updates the waiting passenger count, activates the floor call button,
        and creates a tracking record for the new passenger.

        Args:
            start_floor: The floor where the passenger appears.
            destination_floor: The passenger's destination floor.

        Returns:
            The unique ID assigned to the newly generated passenger.
        """
        # Determine start floor, randomly if not specified
        if start_floor is None:
            start_floor = self.np_random.integers(0, self.num_floors)
        elif not (0 <= start_floor < self.num_floors):
            raise ValueError(f"Invalid start floor: {start_floor}")

        # Determine destination floor, randomly and ensuring it's different from the start
        if destination_floor is None:
            destination_floor = start_floor
            while destination_floor == start_floor:
                destination_floor = self.np_random.integers(0, self.num_floors)
        elif not (0 <= destination_floor < self.num_floors):
            raise ValueError(f"Invalid destination floor: {destination_floor}")
        elif destination_floor == start_floor:
            logger.warning(f"Passenger generated with same start and destination: {start_floor}. This is unusual.")

        # Update state to reflect the new passenger
        self.waiting_passengers[start_floor] += 1
        self.floor_buttons[start_floor] = True

        # Create a unique ID and tracking record for the passenger
        passenger_id = f"{self.current_step}_{start_floor}_{destination_floor}_{self.waiting_passengers[start_floor]}"
        self.waiting_time[passenger_id] = {
            "start_floor": start_floor,
            "destination_floor": destination_floor,
            "wait_start": self.current_step,
            "wait_end": None,       # To be filled upon pickup
            "travel_end": None,     # To be filled upon dropoff
            "elevator_id": None     # To be filled upon pickup
        }
        logger.debug(f"Generated passenger {passenger_id} at floor {start_floor} -> {destination_floor}")
        return passenger_id
    
    def calculate_reward(self) -> float:
        """
        Calculates the total reward for the current step.

        The total reward is a weighted sum of several components designed to encourage
        efficient and timely passenger service. The components are:
        - Movement: Rewards for intelligent elevator movement.
        - Pickup: Rewards for picking up waiting passengers.
        - Delivery: Rewards for dropping off passengers at their destination.
        - Button Response: Rewards for moving towards floors with active call buttons.
        - Time Penalty: A penalty that increases the longer passengers wait.

        Returns:
            The calculated total reward for the current step.
        """
        logger.debug("--- Calculating reward at step %d ---", self.current_step)

        # Calculate each component of the reward
        movement_reward = self._calculate_movement_reward()
        pickup_reward = self._calculate_pickup_reward()
        delivery_reward = self._calculate_delivery_reward()
        button_reward = self._calculate_button_response_reward()
        time_penalty = self._calculate_time_penalty()  # This is a penalty, so it's typically negative

        # Log the raw, unweighted values of each component for debugging
        logger.debug(
            f"Raw Reward Components: Move={movement_reward:.2f}, Pickup={pickup_reward:.2f}, "
            f"Deliver={delivery_reward:.2f}, Button={button_reward:.2f}, TimePen={time_penalty:.2f}"
        )

        # Combine the components using their predefined weights
        total_reward = (
            WEIGHT_MOVEMENT * movement_reward +
            WEIGHT_PICKUP * pickup_reward +
            WEIGHT_DELIVERY * delivery_reward +
            WEIGHT_BUTTON_RESPONSE * button_reward +
            time_penalty  # Time penalty is not weighted, it's a direct penalty
        )

        # Log the final weighted reward for debugging
        logger.debug(f"Total Weighted Reward: {total_reward:.2f}")
        return total_reward

    def _calculate_movement_reward(self) -> float:
        """
        Calculates the reward/penalty based on the movement of each elevator.

        This method assesses whether an elevator's movement is purposeful.
        - It penalizes elevators that are idle while carrying passengers.
        - It rewards elevators for moving towards a passenger's destination.
        - It penalizes moving in a direction with no onboard passengers destined that way.
        - It rewards empty elevators for moving towards floors with waiting passengers.

        Returns:
            The total movement-related reward for the current step.
        """
        movement_reward = 0.0

        for elev_id in range(self.num_elevators):
            elev_pos = self.elevator_positions[elev_id]
            elev_dir = self.elevator_directions[elev_id]
            has_passengers = len(self.passengers_in_elevators[elev_id]) > 0

            if has_passengers:
                # Penalize being idle with passengers onboard
                if elev_dir == ELEVATOR_IDLE:
                    movement_reward += IDLE_WITH_PASSENGERS_PENALTY
                # Reward moving towards a destination
                elif self._is_moving_to_destination(elev_id):
                    movement_reward += MOVING_TO_DESTINATION_BONUS
                # Penalize moving in the wrong direction
                else:
                    movement_reward += self._calculate_direction_penalty(elev_id, elev_pos, elev_dir)
            # Logic for empty elevators
            elif elev_dir != ELEVATOR_IDLE:
                movement_reward += self._calculate_empty_movement_reward(elev_id, elev_pos, elev_dir)

        if movement_reward != 0.0:
            logger.debug(f"Movement reward: {movement_reward:.2f}")
        return movement_reward

    def _calculate_direction_penalty(self, elev_id: int, elev_pos: int, elev_dir: int) -> float:
        """Calculates penalty for an elevator with passengers moving in a non-optimal direction."""
        # Moving up with no passengers who need to go up
        if elev_dir == ELEVATOR_UP:
            has_dest_above = any(
                self.waiting_time[p_id]["destination_floor"] > elev_pos
                for p_id in self.passengers_in_elevators[elev_id] if p_id in self.waiting_time
            )
            # Also check if there are any waiting passengers above to justify the movement
            has_waiting_above = any(self.waiting_passengers[floor] > 0 for floor in range(elev_pos + 1, self.num_floors))
            if not has_dest_above and not has_waiting_above:
                return MOVING_WRONG_DIRECTION_PENALTY

        # Moving down with no passengers who need to go down
        elif elev_dir == ELEVATOR_DOWN:
            has_dest_below = any(
                self.waiting_time[p_id]["destination_floor"] < elev_pos
                for p_id in self.passengers_in_elevators[elev_id] if p_id in self.waiting_time
            )
            # Also check if there are any waiting passengers below
            has_waiting_below = any(self.waiting_passengers[floor] > 0 for floor in range(0, elev_pos))
            if not has_dest_below and not has_waiting_below:
                return MOVING_WRONG_DIRECTION_PENALTY
        return 0.0

    def _calculate_empty_movement_reward(self, elev_id: int, elev_pos: int, elev_dir: int) -> float:
        """Calculates reward for an empty elevator moving towards a call."""
        # Moving up towards a waiting passenger
        if elev_dir == ELEVATOR_UP:
            waiting_floors_above = [f for f in range(elev_pos + 1, self.num_floors) if self.waiting_passengers[f] > 0]
            if waiting_floors_above:
                # Bonus for moving to pickup, could be enhanced to check for closer elevators
                return MOVING_TO_PICKUP_BONUS
            return MOVING_WITHOUT_PASSENGERS_PENALTY

        # Moving down towards a waiting passenger
        elif elev_dir == ELEVATOR_DOWN:
            waiting_floors_below = [f for f in range(0, elev_pos) if self.waiting_passengers[f] > 0]
            if waiting_floors_below:
                return MOVING_TO_PICKUP_BONUS
            return MOVING_WITHOUT_PASSENGERS_PENALTY

        return 0.0

    def _calculate_pickup_reward(self) -> float:
        """
        Calculates the reward for picking up passengers.

        A reward is given for each passenger picked up during the current step.
        The reward is scaled by how long the passenger has been waiting, incentivizing
        the agent to prioritize passengers who have waited longer.

        Returns:
            The total pickup-related reward for the current step.
        """
        pickup_reward = 0.0
        for passenger_id, data in self.waiting_time.items():
            # Check if the passenger was picked up in the current step
            if data["wait_end"] == self.current_step:
                waiting_duration = data["wait_end"] - data["wait_start"]
                # The reward is the base amount plus a bonus proportional to the waiting time
                reward_multiplier = min(MAX_PICKUP_MULTIPLIER, 1.0 + PICKUP_WAIT_MULTIPLIER * waiting_duration)
                pickup_reward += PICKUP_BASE_REWARD * reward_multiplier

        if pickup_reward != 0.0:
            logger.debug(f"Pickup reward: {pickup_reward:.2f}")
        return pickup_reward

    def _calculate_delivery_reward(self) -> float:
        """
        Calculates the reward for delivering passengers to their destinations.

        A significant reward is given for each successful delivery. The reward is enhanced
        by bonuses for speed and efficiency, comparing the actual travel time to an
        expected travel time.

        Returns:
            The total delivery-related reward for the current step.
        """
        delivery_reward = 0.0
        for passenger_id, data in self.waiting_time.items():
            # Check if the passenger was delivered in the current step
            if data["travel_end"] == self.current_step and passenger_id in self.waiting_time:
                # Ensure data integrity before calculation
                if data["wait_end"] is None or data["start_floor"] is None:
                    logger.warning(f"Incomplete data for delivered passenger {passenger_id}. Skipping reward.")
                    continue

                # Base reward for any successful delivery
                delivery_reward += DELIVERY_BASE_REWARD

                # Calculate time-based bonuses
                wait_time = data["wait_end"] - data["wait_start"]
                travel_time = data["travel_end"] - data["wait_end"]
                floor_distance = abs(data["destination_floor"] - data["start_floor"])
                expected_travel_time = floor_distance + 1  # Simple model for expected time

                # Bonus for efficiency (less travel time than expected)
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
        Calculates a penalty based on the cumulative waiting time of all passengers.

        A penalty is applied for each step that a passenger is waiting to be picked up.
        This penalty grows over time to discourage leaving passengers waiting for too long.

        Returns:
            The total time-based penalty for the current step (a negative value).
        """
        time_penalty = 0.0
        for passenger_id, data in self.waiting_time.items():
            # Apply penalty only to passengers who are still waiting
            if data["wait_end"] is None:
                waiting_duration = self.current_step - data["wait_start"]
                penalty = TIME_PENALTY_BASE * (1.0 + TIME_PENALTY_MULTIPLIER * waiting_duration)
                # Cap the penalty to avoid excessively large negative rewards
                time_penalty += max(MAX_TIME_PENALTY, penalty)

        if time_penalty != 0.0:
            logger.debug(f"Time penalty: {time_penalty:.2f}")
        return time_penalty

    def _calculate_button_response_reward(self) -> float:
        """
        Calculates a reward for elevators moving towards floors with active call buttons.

        This encourages the agent to be responsive to new passenger requests, especially
        those that have been waiting for a long time.

        Returns:
            The total button-response-related reward for the current step.
        """
        button_reward = 0.0
        for floor in range(self.num_floors):
            if self.floor_buttons[floor]:
                # Find the longest waiting passenger on this floor
                longest_wait = max((
                    self.current_step - data["wait_start"]
                    for data in self.waiting_time.values()
                    if data["start_floor"] == floor and data["wait_end"] is None
                ), default=0)

                # If a passenger has been waiting for a while, reward responsive elevators
                if longest_wait > LONG_WAIT_THRESHOLD:
                    for elev_id in range(self.num_elevators):
                        elev_pos = self.elevator_positions[elev_id]
                        elev_dir = self.elevator_directions[elev_id]
                        # Check if elevator is moving towards this floor
                        if (elev_dir == ELEVATOR_UP and floor > elev_pos) or \
                           (elev_dir == ELEVATOR_DOWN and floor < elev_pos):
                            reward_multiplier = min(MAX_BUTTON_RESPONSE_MULTIPLIER, 1.0 + BUTTON_RESPONSE_WAIT_MULTIPLIER * longest_wait)
                            button_reward += BUTTON_RESPONSE_BASE * reward_multiplier

        if button_reward != 0.0:
            logger.debug(f"Button response reward: {button_reward:.2f}")
        return button_reward

    def _is_moving_to_destination(self, elevator_id: int) -> bool:
        """
        Checks if an elevator is moving towards the destination of one of its passengers.

        Args:
            elevator_id: The ID of the elevator to check.

        Returns:
            True if the elevator is moving towards a passenger's destination, False otherwise.
        """
        curr_pos = self.elevator_positions[elevator_id]
        curr_dir = self.elevator_directions[elevator_id]

        if curr_dir == ELEVATOR_IDLE:
            return False

        for passenger_id in self.passengers_in_elevators[elevator_id]:
            if passenger_id not in self.waiting_time:
                continue  # Skip if passenger data is somehow missing

            dest_floor = self.waiting_time[passenger_id]["destination_floor"]
            if (curr_dir == ELEVATOR_UP and dest_floor > curr_pos) or \
               (curr_dir == ELEVATOR_DOWN and dest_floor < curr_pos):
                return True
        return False

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment.

        This involves advancing the simulation by one step, which includes:
        1. Processing the agent's action for each elevator.
        2. Generating new passengers (either randomly or from a scenario).
        3. Calculating the reward for the action taken.
        4. Checking for episode termination or truncation.
        5. Optionally rendering the environment.

        Args:
            action: A numpy array containing the action for each elevator.

        Returns:
            A tuple containing the new observation, reward, termination flag,
            truncation flag, and auxiliary information.
        """
        self.current_step += 1

        # Apply the chosen action to each elevator
        for elev_id in range(self.num_elevators):
            self._process_elevator_action(elev_id, action[elev_id])

        # Generate new passengers, either from a predefined scenario or randomly
        if self.scenario:
            # In scenario mode, add passengers based on their specified arrival step
            while (self.scenario_index < len(self.scenario) and
                   self.scenario[self.scenario_index]['arrival_step'] == self.current_step):
                passenger_data = self.scenario[self.scenario_index]
                self._generate_passenger(
                    start_floor=passenger_data['start_floor'],
                    destination_floor=passenger_data['destination_floor']
                )
                self.scenario_index += 1
        else:
            # In random mode, generate passengers based on the passenger rate
            if self.np_random.random() < self.passenger_rate:
                self._generate_passenger()

        # Calculate the reward for the state transition
        reward = self.calculate_reward()
        self.current_reward = reward

        # Check for termination (episode ends due to reaching max steps)
        terminated = self.current_step >= self.max_steps
        truncated = False  # Truncation is not used in this version

        # Render the environment if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _process_elevator_action(self, elevator_id: int, action: int) -> None:
        """
        Processes a single elevator's action for the current step.

        This method updates the elevator's position and direction based on the
        chosen action. After moving, it triggers passenger interactions at the new floor.

        Args:
            elevator_id: The ID of the elevator to process.
            action: The action to take (0: idle, 1: up, 2: down).
        """
        elev_pos = self.elevator_positions[elevator_id]

        # Update elevator position based on the action, respecting floor boundaries
        if action == ELEVATOR_UP and elev_pos < self.num_floors - 1:
            self.elevator_positions[elevator_id] += 1
            self.elevator_directions[elevator_id] = ELEVATOR_UP
        elif action == ELEVATOR_DOWN and elev_pos > 0:
            self.elevator_positions[elevator_id] -= 1
            self.elevator_directions[elevator_id] = ELEVATOR_DOWN
        else:
            # If action is idle, or at top/bottom floor, set direction to idle
            self.elevator_directions[elevator_id] = ELEVATOR_IDLE

        # Handle passenger drop-offs and pickups at the new floor
        self._process_floor(elevator_id)

    def _process_floor(self, elevator_id: int) -> None:
        """
        Manages passenger interactions (drop-off and pickup) at an elevator's current floor.

        This is a two-step process: first, passengers destined for the current floor are
        dropped off. Second, passengers waiting on the current floor are picked up.

        Args:
            elevator_id: The ID of the elevator at the floor.
        """
        current_floor = self.elevator_positions[elevator_id]
        self._process_dropoff(elevator_id, current_floor)
        self._process_pickup(elevator_id, current_floor)

    def _process_dropoff(self, elevator_id: int, current_floor: int) -> None:
        """
        Handles the drop-off of passengers from an elevator at the current floor.

        It checks for any passengers inside the elevator whose destination is the
        current floor, removes them, and updates their status and simulation statistics.

        Args:
            elevator_id: The ID of the elevator.
            current_floor: The floor where the elevator is located.
        """
        # Identify passengers whose destination is the current floor
        passengers_to_remove = [
            p_id for p_id in self.passengers_in_elevators[elevator_id]
            if self.waiting_time[p_id]["destination_floor"] == current_floor
        ]

        for passenger_id in passengers_to_remove:
            # Remove passenger from the elevator
            self.passengers_in_elevators[elevator_id].remove(passenger_id)

            # Mark their travel as complete and update statistics
            self.waiting_time[passenger_id]["travel_end"] = self.current_step
            self.total_waiting_time += (self.waiting_time[passenger_id]["travel_end"] - self.waiting_time[passenger_id]["wait_start"])
            self.total_passengers_served += 1

        # Deactivate the elevator's internal button for this floor if no more passengers are going here
        if not any(self.waiting_time[p]["destination_floor"] == current_floor for p in self.passengers_in_elevators[elevator_id]):
            self.elevator_buttons[current_floor, elevator_id] = False

    def _process_pickup(self, elevator_id: int, current_floor: int) -> None:
        """
        Handles the pickup of waiting passengers at the current floor.

        It checks for waiting passengers on the floor, moves them into the elevator,
        updates their status, and activates the corresponding destination button inside the elevator.

        Args:
            elevator_id: The ID of the elevator.
            current_floor: The floor where the elevator is located.
        """
        if self.waiting_passengers[current_floor] > 0:
            # Identify all passengers waiting on this floor
            passengers_to_pickup = [
                p_id for p_id, data in self.waiting_time.items()
                if data["start_floor"] == current_floor and data["wait_end"] is None
            ]

            for passenger_id in passengers_to_pickup:
                # Add passenger to the elevator
                self.passengers_in_elevators[elevator_id].append(passenger_id)

                # Mark them as picked up and record which elevator they entered
                self.waiting_time[passenger_id]["wait_end"] = self.current_step
                self.waiting_time[passenger_id]["elevator_id"] = elevator_id

                # Press the destination button inside the elevator
                dest_floor = self.waiting_time[passenger_id]["destination_floor"]
                self.elevator_buttons[dest_floor, elevator_id] = True
                self.waiting_passengers[current_floor] -= 1

            # Deactivate the floor call button if no one is left waiting
            if self.waiting_passengers[current_floor] == 0:
                self.floor_buttons[current_floor] = False
    
    def _render_setup(self) -> None:
        """Initializes the Pygame window and font for rendering."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Multi-Elevator Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

    def _render_frame(self) -> None:
        """
        Renders a single frame of the environment to the screen.

        This method draws all visual elements, including floors, elevators,
        passengers, and status information.
        """
        if self.screen is None:
            self._render_setup()

        self.screen.fill(COLORS['WHITE'])

        # Draw all components of the simulation
        self._draw_floors(FLOOR_HEIGHT, Y_OFFSET)
        self._draw_elevators(ELEVATOR_SPACING, ELEVATOR_WIDTH, FLOOR_HEIGHT, FLOOR_HEIGHT, Y_OFFSET)
        self._draw_info()

        # Update the display and control the frame rate
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_floors(self, floor_height: int, y_offset: int) -> None:
        """
        Draws the floors, call buttons, and waiting passenger counts.

        Args:
            floor_height: The height of each floor in pixels.
            y_offset: The vertical offset for drawing.
        """
        for floor in range(self.num_floors):
            y = SCREEN_HEIGHT - (floor + 1) * floor_height - y_offset

            # Draw the line representing the floor
            pygame.draw.line(self.screen, COLORS['GRAY'], (0, y + floor_height), (SCREEN_WIDTH, y + floor_height), 2)

            # Draw the floor number label
            text = self.font.render(f"Floor {floor}", True, COLORS['BLACK'])
            self.screen.blit(text, (10, y + floor_height // 2 - 10))

            # Draw the call button (green if active, red if not)
            button_color = COLORS['GREEN'] if self.floor_buttons[floor] else COLORS['RED']
            pygame.draw.circle(self.screen, button_color, (SCREEN_WIDTH - 30, y + floor_height // 2), 12)

            # Display the number of waiting passengers
            if self.waiting_passengers[floor] > 0:
                wait_text = self.font.render(f"Waiting: {self.waiting_passengers[floor]}", True, COLORS['BLACK'])
                self.screen.blit(wait_text, (SCREEN_WIDTH - 150, y + floor_height // 2 - 10))

    def _draw_elevators(self, elevator_spacing: int, elevator_width: int,
                       elevator_height: int, floor_height: int, y_offset: int) -> None:
        """
        Draws the elevators, their occupants, and direction indicators.

        Args:
            elevator_spacing: The horizontal spacing between elevators.
            elevator_width: The width of each elevator.
            elevator_height: The height of each elevator.
            floor_height: The height of each floor.
            y_offset: The vertical offset for drawing.
        """
        for elev_id in range(self.num_elevators):
            # Calculate the elevator's screen position
            elevator_x = 100 + elev_id * (elevator_width + elevator_spacing)
            elevator_y = SCREEN_HEIGHT - (self.elevator_positions[elev_id] + 1) * floor_height - y_offset

            # Draw the elevator car
            pygame.draw.rect(self.screen, COLORS['BLUE'], (elevator_x, elevator_y, elevator_width, elevator_height))

            # Display elevator ID and number of passengers
            passenger_count = len(self.passengers_in_elevators[elev_id])
            text = self.font.render(f"E{elev_id} (P:{passenger_count})", True, COLORS['WHITE'])
            text_rect = text.get_rect(center=(elevator_x + elevator_width // 2, elevator_y + 20))
            self.screen.blit(text, text_rect)

            # Display the direction indicator (up, down, or idle)
            direction_symbol = "↑" if self.elevator_directions[elev_id] == ELEVATOR_UP else \
                             "↓" if self.elevator_directions[elev_id] == ELEVATOR_DOWN else "•"
            dir_text = self.font.render(direction_symbol, True, COLORS['WHITE'])
            dir_rect = dir_text.get_rect(center=(elevator_x + elevator_width // 2, elevator_y + 40))
            self.screen.blit(dir_text, dir_rect)

    def _draw_info(self) -> None:
        """Draws simulation status information at the top of the screen."""
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)

        info_texts = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Served: {self.total_passengers_served}",
            f"Avg Wait: {avg_wait_time:.2f}",
            f"Reward: {self.current_reward:.2f}"
        ]

        for i, text in enumerate(info_texts):
            surf = self.font.render(text, True, COLORS['BLACK'])
            self.screen.blit(surf, (10 + i * 180, 10))

    def close(self) -> None:
        """Cleans up the environment, primarily by closing the Pygame window."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
