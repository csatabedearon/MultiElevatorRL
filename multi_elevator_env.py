import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
import random
from collections import deque
import logging
from typing import Dict, Any, Optional, List, Tuple

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MultiElevatorEnv(gym.Env):
    """
    An extended elevator environment with configurable number of elevators and floors.
    The goal is to minimize passenger waiting time.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000}

    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 500,
                 passenger_rate: float = 0.1, num_elevators: int = 3, num_floors: int = 5) -> None:
        super().__init__()
        self.num_floors = num_floors
        self.num_elevators = num_elevators
        self.max_steps = max_steps
        self.passenger_rate = passenger_rate
        
        # Observation space
        self.observation_space = spaces.Dict({
            "elevator_positions": spaces.MultiDiscrete([self.num_floors] * self.num_elevators),
            "elevator_directions": spaces.MultiDiscrete([3] * self.num_elevators),  # 0: idle, 1: up, 2: down
            "elevator_buttons": spaces.MultiBinary([self.num_floors, self.num_elevators]),
            "floor_buttons": spaces.MultiBinary(self.num_floors),
            "waiting_passengers": spaces.Box(low=0, high=100, shape=(self.num_floors,), dtype=np.int32)
        })
        
        # Action space: For each elevator: 0 (idle), 1 (up), 2 (down)
        self.action_space = spaces.MultiDiscrete([3] * self.num_elevators)
        
        # Rendering variables
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.colors = None
        self.font = None

        self.current_reward: float = 0.0
        
        # Reset environment to initialize state variables
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
        """Reset the environment and all state variables."""
        super().reset(seed=seed)
        logger.debug("Resetting MultiElevatorEnv")
        
        self.elevator_positions = np.zeros(self.num_elevators, dtype=np.int32)
        self._distribute_elevators()
        self.elevator_directions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_buttons = np.zeros((self.num_floors, self.num_elevators), dtype=bool)
        self.floor_buttons = np.zeros(self.num_floors, dtype=bool)
        self.waiting_passengers = np.zeros(self.num_floors, dtype=np.int32)
        self.passengers_in_elevators: List[List[str]] = [[] for _ in range(self.num_elevators)]
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        self.waiting_time: Dict[str, Dict[str, Any]] = {}
        self.current_reward = 0.0
        
        # Generate initial passengers
        for _ in range(5):
            self._generate_passenger()
        
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
        """Calculate and return the total reward based on movement, pickup, delivery, time penalty, and button response."""
        movement_reward = self._calculate_movement_reward()
        pickup_reward = self._calculate_pickup_reward()
        delivery_reward = self._calculate_delivery_reward()
        time_penalty = self._calculate_time_penalty()
        button_reward = self._calculate_button_response_reward()
        
        total_reward = (0.3 * movement_reward +
                        0.2 * pickup_reward +
                        0.3 * delivery_reward +
                        0.1 * button_reward +
                        time_penalty)
        logger.debug("Calculated total reward: %.2f", total_reward)
        return total_reward

    def _calculate_button_response_reward(self) -> float:
        button_reward = 0.0
        for floor in range(self.num_floors):
            if self.floor_buttons[floor]:
                longest_wait = 0
                for passenger_id, data in self.waiting_time.items():
                    if data["start_floor"] == floor and data["wait_end"] is None:
                        wait_duration = self.current_step - data["wait_start"]
                        longest_wait = max(longest_wait, wait_duration)
                if longest_wait > 5:
                    for elev_id in range(self.num_elevators):
                        elev_pos = self.elevator_positions[elev_id]
                        elev_dir = self.elevator_directions[elev_id]
                        if (elev_dir == 1 and floor > elev_pos) or (elev_dir == 2 and floor < elev_pos):
                            button_reward += 0.1 * min(5.0, 1.0 + 0.1 * longest_wait)
        return button_reward

    def _calculate_movement_reward(self) -> float:
        movement_reward = 0.0
        for elev_id in range(self.num_elevators):
            elev_pos = self.elevator_positions[elev_id]
            elev_dir = self.elevator_directions[elev_id]
            has_passengers = len(self.passengers_in_elevators[elev_id]) > 0
            if elev_dir == 0 and has_passengers:
                movement_reward -= 1.0
            if has_passengers:
                if self._is_moving_to_destination(elev_id) and elev_dir != 0:
                    movement_reward += 0.5
                elif elev_dir == 1:
                    has_dest_above = any(
                        (p_id in self.waiting_time and self.waiting_time[p_id]["destination_floor"] > elev_pos)
                        for p_id in self.passengers_in_elevators[elev_id]
                    )
                    if not has_dest_above and not any(self.waiting_passengers[floor] > 0 for floor in range(elev_pos + 1, self.num_floors)):
                        movement_reward -= 0.8
                elif elev_dir == 2:
                    has_dest_below = any(
                        (p_id in self.waiting_time and self.waiting_time[p_id]["destination_floor"] < elev_pos)
                        for p_id in self.passengers_in_elevators[elev_id]
                    )
                    if not has_dest_below and not any(self.waiting_passengers[floor] > 0 for floor in range(0, elev_pos)):
                        movement_reward -= 0.8
            elif not has_passengers and elev_dir != 0:
                if elev_dir == 1:
                    waiting_floors_above = [floor for floor in range(elev_pos + 1, self.num_floors) if self.waiting_passengers[floor] > 0]
                    if waiting_floors_above:
                        closest_floor = min(waiting_floors_above)
                        other_elevators_heading_up = [i for i in range(self.num_elevators)
                                                      if i != elev_id and self.elevator_directions[i] == 1 and self.elevator_positions[i] < closest_floor]
                        if not other_elevators_heading_up or all(self.elevator_positions[i] <= elev_pos for i in other_elevators_heading_up):
                            movement_reward += 0.3
                        else:
                            movement_reward += 0.1
                    else:
                        movement_reward -= 0.5
                elif elev_dir == 2:
                    waiting_floors_below = [floor for floor in range(0, elev_pos) if self.waiting_passengers[floor] > 0]
                    if waiting_floors_below:
                        closest_floor = max(waiting_floors_below)
                        other_elevators_heading_down = [i for i in range(self.num_elevators)
                                                        if i != elev_id and self.elevator_directions[i] == 2 and self.elevator_positions[i] > closest_floor]
                        if not other_elevators_heading_down or all(self.elevator_positions[i] >= elev_pos for i in other_elevators_heading_down):
                            movement_reward += 0.3
                        else:
                            movement_reward += 0.1
                    else:
                        movement_reward -= 0.5
        return movement_reward

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

    def _calculate_pickup_reward(self) -> float:
        pickup_reward = 0.0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] == self.current_step:
                waiting_duration = data["wait_end"] - data["wait_start"]
                pickup_reward += 2.0 * min(5.0, 1.0 + 0.2 * waiting_duration)
        return pickup_reward

    def _calculate_delivery_reward(self) -> float:
        delivery_reward = 0.0
        for passenger_id, data in self.waiting_time.items():
            if data["travel_end"] == self.current_step:
                delivery_reward += 15.0
                wait_time = data["wait_end"] - data["wait_start"]
                travel_time = data["travel_end"] - data["wait_end"]
                start_floor = data["start_floor"]
                dest_floor = data["destination_floor"]
                floor_distance = abs(dest_floor - start_floor)
                expected_travel_time = floor_distance + 1
                time_efficiency = max(0, expected_travel_time - travel_time)
                time_bonus = 10.0 * (0.9 ** max(0, travel_time - expected_travel_time)) \
                             + 5.0 * (0.9 ** wait_time) + 0.5 * time_efficiency
                delivery_reward += time_bonus
        return delivery_reward

    def _calculate_time_penalty(self) -> float:
        time_penalty = 0.0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] is None:
                wait_duration = self.current_step - data["wait_start"]
                if wait_duration <= 5:
                    time_penalty -= 0.1 * wait_duration
                else:
                    time_penalty -= 0.1 * 5 + 0.3 * (wait_duration - 5) + 0.1 * (1.1 ** min(wait_duration - 5, 20))
        for elev_id in range(self.num_elevators):
            for passenger_id in self.passengers_in_elevators[elev_id]:
                if passenger_id not in self.waiting_time:
                    continue
                data = self.waiting_time[passenger_id]
                if data["wait_end"] is None:
                    continue
                travel_duration = self.current_step - data["wait_end"]
                start_floor = data["start_floor"]
                dest_floor = data["destination_floor"]
                floor_distance = abs(dest_floor - start_floor)
                expected_travel_time = floor_distance + 1
                excess_travel_time = max(0, travel_duration - expected_travel_time)
                if excess_travel_time > 0:
                    time_penalty -= 0.2 * excess_travel_time + 0.1 * (1.1 ** min(excess_travel_time, 30))
        return time_penalty

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self.current_step += 1
        for elev_id in range(self.num_elevators):
            self._process_elevator_action(elev_id, action[elev_id])
        if self.np_random.random() < self.passenger_rate:
            self._generate_passenger()
        reward = self.calculate_reward()
        self.current_reward = reward
        terminated = self.current_step >= self.max_steps
        truncated = False
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _process_elevator_action(self, elevator_id: int, action: int) -> None:
        elev_pos = self.elevator_positions[elevator_id]
        if action == 1 and elev_pos < self.num_floors - 1:
            self.elevator_positions[elevator_id] += 1
            self.elevator_directions[elevator_id] = 1
        elif action == 2 and elev_pos > 0:
            self.elevator_positions[elevator_id] -= 1
            self.elevator_directions[elevator_id] = 2
        else:
            self.elevator_directions[elevator_id] = 0
        self._process_floor(elevator_id)
    
    def _process_floor(self, elevator_id: int) -> None:
        current_floor = self.elevator_positions[elevator_id]
        self._process_dropoff(elevator_id, current_floor)
        self._process_pickup(elevator_id, current_floor)
    
    def _process_dropoff(self, elevator_id: int, current_floor: int) -> None:
        passengers_to_remove = []
        for passenger_id in self.passengers_in_elevators[elevator_id]:
            data = self.waiting_time[passenger_id]
            if data["destination_floor"] == current_floor:
                passengers_to_remove.append(passenger_id)
        for passenger_id in passengers_to_remove:
            self.passengers_in_elevators[elevator_id].remove(passenger_id)
            self.waiting_time[passenger_id]["travel_end"] = self.current_step
            self.total_waiting_time += (self.waiting_time[passenger_id]["travel_end"] -
                                        self.waiting_time[passenger_id]["wait_start"])
            self.total_passengers_served += 1
        if not any(self.waiting_time[p]["destination_floor"] == current_floor
                   for p in self.passengers_in_elevators[elevator_id]):
            self.elevator_buttons[current_floor, elevator_id] = False
    
    def _process_pickup(self, elevator_id: int, current_floor: int) -> None:
        if self.waiting_passengers[current_floor] > 0:
            passengers_to_pickup = []
            for passenger_id, data in self.waiting_time.items():
                if data["start_floor"] == current_floor and data["wait_end"] is None and data["elevator_id"] is None:
                    passengers_to_pickup.append(passenger_id)
            for passenger_id in passengers_to_pickup:
                self.passengers_in_elevators[elevator_id].append(passenger_id)
                self.waiting_time[passenger_id]["wait_end"] = self.current_step
                self.waiting_time[passenger_id]["elevator_id"] = elevator_id
                dest_floor = self.waiting_time[passenger_id]["destination_floor"]
                self.elevator_buttons[dest_floor, elevator_id] = True
                self.waiting_passengers[current_floor] -= 1
            if self.waiting_passengers[current_floor] == 0:
                self.floor_buttons[current_floor] = False
            else:
                self.floor_buttons[current_floor] = True
    
    def _render_setup(self) -> None:
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Multi-Elevator Simulator")
        self.clock = pygame.time.Clock()
        self.colors = {
            "background": (240, 240, 240),
            "elevator": [(100, 100, 100), (120, 80, 80), (80, 120, 80), (80, 80, 120)],
            "floor": (200, 200, 200),
            "text": (0, 0, 0),
            "button_off": (150, 150, 150),
            "button_on": (255, 0, 0),
        }
        self.font = pygame.font.SysFont(None, 24)
        self.button_font = pygame.font.SysFont(None, 12)
    
    def _render_frame(self) -> None:
        if self.screen is None:
            self._render_setup()
        self.screen.fill(self.colors["background"])
        info_height = 80
        available_height = self.screen_height - info_height
        floor_height = available_height // self.num_floors
        elevator_width = 60
        elevator_height = floor_height - 10
        elevator_spacing = self.screen_width // (self.num_elevators + 2)
        self._draw_floors(floor_height, info_height)
        self._draw_elevators(elevator_spacing, elevator_width, elevator_height, floor_height, info_height)
        self._draw_info()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        pygame.event.pump()
    
    def _draw_floors(self, floor_height: int, y_offset: int) -> None:
        for floor in range(self.num_floors):
            y = self.screen_height - (floor + 1) * floor_height - y_offset
            pygame.draw.line(self.screen, self.colors["floor"],
                             (0, y + floor_height), (self.screen_width, y + floor_height), 2)
            text = self.font.render(f"Floor {floor}", True, self.colors["text"])
            text_rect = text.get_rect(midleft=(10, y + floor_height // 2))
            self.screen.blit(text, text_rect)
            button_color = self.colors["button_on"] if self.floor_buttons[floor] else self.colors["button_off"]
            pygame.draw.circle(self.screen, button_color,
                               (self.screen_width - 30, y + floor_height // 2), 12)
            if self.waiting_passengers[floor] > 0:
                text = self.font.render(f"Waiting: {self.waiting_passengers[floor]}", True, self.colors["text"])
                text_rect = text.get_rect(midright=(self.screen_width - 50, y + floor_height // 2))
                self.screen.blit(text, text_rect)
    
    def _draw_elevators(self, elevator_spacing: int, elevator_width: int, elevator_height: int,
                        floor_height: int, y_offset: int) -> None:
        for elev_id in range(self.num_elevators):
            elevator_x = (elev_id + 1) * elevator_spacing - elevator_width // 2
            elevator_y = self.screen_height - (self.elevator_positions[elev_id] + 1) * floor_height - y_offset
            elevator_color = self.colors["elevator"][min(elev_id, len(self.colors["elevator"]) - 1)]
            pygame.draw.rect(self.screen, elevator_color,
                             (elevator_x, elevator_y, elevator_width, elevator_height))
            text = self.font.render(f"E{elev_id} (P:{len(self.passengers_in_elevators[elev_id])})", True, (255, 255, 255))
            text_rect = text.get_rect(center=(elevator_x + elevator_width // 2, elevator_y + elevator_height // 2))
            self.screen.blit(text, text_rect)
            direction_symbol = "↑" if self.elevator_directions[elev_id] == 1 else "↓" if self.elevator_directions[elev_id] == 2 else "•"
            dir_text = self.font.render(direction_symbol, True, (255, 255, 255))
            dir_rect = dir_text.get_rect(center=(elevator_x + elevator_width // 2, elevator_y + elevator_height - 15))
            self.screen.blit(dir_text, dir_rect)
    
    def _draw_info(self) -> None:
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)
        info_text = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Total served: {self.total_passengers_served}",
            f"Avg wait time: {avg_wait_time:.2f} steps",
            f"Reward: {self.current_reward:.2f}"
        ]
        for i, text in enumerate(info_text):
            surf = self.font.render(text, True, self.colors["text"])
            rect = surf.get_rect(topleft=(10 + i * (self.screen_width // len(info_text)), 10))
            self.screen.blit(surf, rect)
    
    def close(self) -> None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Register the environment for Gym usage
from gymnasium.envs.registration import register
register(
    id='MultiElevator-v0',
    entry_point='multi_elevator_env:MultiElevatorEnv',
)