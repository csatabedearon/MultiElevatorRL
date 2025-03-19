import gymnasium as gym
import pygame
import numpy as np
from gymnasium import spaces
import random
from collections import deque

class MultiElevatorEnv(gym.Env):
    """
    An extended elevator environment with multiple elevators and 10 floors.
    The goal is to minimize passenger waiting time.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000}

    def __init__(self, render_mode=None, max_steps=500, passenger_rate=0.1, num_elevators=3):
        self.num_floors = 10  # 10 floors in total
        self.num_elevators = num_elevators
        self.max_steps = max_steps
        self.passenger_rate = passenger_rate  # probability of new passenger each step
        
        # Observation space: 
        # For each elevator: [elevator_position, direction, elevator_buttons]
        # Plus global floor_buttons and waiting_passengers
        self.observation_space = spaces.Dict({
            "elevator_positions": spaces.MultiDiscrete([self.num_floors] * self.num_elevators),
            "elevator_directions": spaces.MultiDiscrete([3] * self.num_elevators),  # 0: idle, 1: up, 2: down
            "elevator_buttons": spaces.MultiBinary([self.num_floors, self.num_elevators]),
            "floor_buttons": spaces.MultiBinary(self.num_floors),
            "waiting_passengers": spaces.Box(low=0, high=100, shape=(self.num_floors,), dtype=np.int32)
        })
        
        # Action space: For each elevator: 0 (idle), 1 (up), 2 (down)
        self.action_space = spaces.MultiDiscrete([3] * self.num_elevators)
        
        # Initialize environment state
        self.elevator_positions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_directions = np.zeros(self.num_elevators, dtype=np.int32)  # 0: idle, 1: up, 2: down
        self.elevator_buttons = np.zeros((self.num_floors, self.num_elevators), dtype=bool)
        self.floor_buttons = np.zeros(self.num_floors, dtype=bool)
        self.waiting_passengers = np.zeros(self.num_floors, dtype=np.int32)
        self.passengers_in_elevators = [[] for _ in range(self.num_elevators)]
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        
        # For visualization
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.elevator_speed = 1  # How many floors per step
        self.waiting_time = {}  # Dict to track waiting time for each passenger
        
    def _get_obs(self):
        return {
            "elevator_positions": self.elevator_positions,
            "elevator_directions": self.elevator_directions,
            "elevator_buttons": self.elevator_buttons,
            "floor_buttons": self.floor_buttons,
            "waiting_passengers": self.waiting_passengers
        }
    
    def _get_info(self):
        passengers_in_elevators_count = sum(len(passengers) for passengers in self.passengers_in_elevators)
        return {
            "waiting_passengers": np.sum(self.waiting_passengers),
            "passengers_in_elevators": passengers_in_elevators_count,
            "average_waiting_time": self.total_waiting_time / max(1, self.total_passengers_served),
            "total_passengers_served": self.total_passengers_served,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment state
        # Place elevators at different starting floors for better coverage
        self.elevator_positions = np.zeros(self.num_elevators, dtype=np.int32)
        if self.num_elevators > 1:
            # Distribute elevators - one at bottom, one at top, others evenly spaced
            self.elevator_positions[0] = 0
            self.elevator_positions[-1] = self.num_floors - 1
            for i in range(1, self.num_elevators - 1):
                self.elevator_positions[i] = (i * self.num_floors) // self.num_elevators
        
        self.elevator_directions = np.zeros(self.num_elevators, dtype=np.int32)
        self.elevator_buttons = np.zeros((self.num_floors, self.num_elevators), dtype=bool)
        self.floor_buttons = np.zeros(self.num_floors, dtype=bool)
        self.waiting_passengers = np.zeros(self.num_floors, dtype=np.int32)
        self.passengers_in_elevators = [[] for _ in range(self.num_elevators)]
        self.current_step = 0
        self.total_waiting_time = 0
        self.total_passengers_served = 0
        self.waiting_time = {}
        
        # Generate initial passengers
        for _ in range(5):  # Start with a few passengers
            self._generate_passenger()
        
        # Initialize renderer if needed
        if self.render_mode == "human":
            self._render_setup()
            
        return self._get_obs(), self._get_info()
    
    def _generate_passenger(self):
        """Generate a new passenger with random start and destination floors"""
        start_floor = self.np_random.integers(0, self.num_floors)
        destination_floor = start_floor
        while destination_floor == start_floor:
            destination_floor = self.np_random.integers(0, self.num_floors)
        
        # Add passenger to waiting queue
        self.waiting_passengers[start_floor] += 1
        self.floor_buttons[start_floor] = True
        
        # Track passenger for waiting time calculation
        passenger_id = f"{self.current_step}_{start_floor}_{destination_floor}"
        self.waiting_time[passenger_id] = {
            "start_floor": start_floor,
            "destination_floor": destination_floor,
            "wait_start": self.current_step,
            "wait_end": None,
            "travel_end": None,
            "elevator_id": None  # Which elevator picked up this passenger
        }
        
        return passenger_id
    
    def calculate_reward(self):
        """Calculate reward balancing passenger waiting time and movement efficiency"""
        # Base reward
        reward = -0.1
        
        # Movement penalty - increased from V6 but not as strict as V7
        movement_penalty = 0
        for elev_id in range(self.num_elevators):
            if self.elevator_directions[elev_id] != 0:
                # Check if elevator has passengers or is moving toward waiting passengers
                has_passengers = len(self.passengers_in_elevators[elev_id]) > 0
                moving_to_pickup = False
                
                # Check if moving toward a floor with waiting passengers
                curr_pos = self.elevator_positions[elev_id]
                curr_dir = self.elevator_directions[elev_id]
                
                if curr_dir == 1:  # Moving up
                    for floor in range(curr_pos + 1, self.num_floors):
                        if self.waiting_passengers[floor] > 0:
                            moving_to_pickup = True
                            break
                elif curr_dir == 2:  # Moving down
                    for floor in range(curr_pos - 1, -1, -1):
                        if self.waiting_passengers[floor] > 0:
                            moving_to_pickup = True
                            break
                
                # Apply penalty with different weights based on purpose
                if has_passengers or moving_to_pickup:
                    movement_penalty -= 0.2  # Reduced penalty for purposeful movement
                else:
                    movement_penalty -= 0.4  # Higher penalty for "speculative" movement
        
        # Pickup reward with slight emphasis on fast pickups
        pickup_reward = 0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] == self.current_step:
                waiting_duration = data["wait_end"] - data["wait_start"]
                # Exponentially decreasing reward based on wait time
                pickup_reward += 7.0 * (0.9 ** waiting_duration)
        
        # Delivery reward with significant bonus for quick service
        delivery_reward = 0
        for passenger_id, data in self.waiting_time.items():
            if data["travel_end"] == self.current_step:
                delivery_reward += 10.0
                total_trip_time = data["travel_end"] - data["wait_start"]
                # More generous time bonus
                time_bonus = 15.0 * (0.95 ** total_trip_time)
                delivery_reward += time_bonus
        
        # Strategic positioning reward - encourage elevators to distribute across floors
        positioning_reward = 0
        positions = self.elevator_positions.copy()
        positions.sort()
        
        # Reward for maintaining good spacing between elevators
        for i in range(1, len(positions)):
            gap = positions[i] - positions[i-1]
            ideal_gap = self.num_floors / self.num_elevators
            # Reward for approaching ideal gap
            if abs(gap - ideal_gap) <= 2:
                positioning_reward += 0.2
        
        # Waiting time penalty with progressive scaling
        waiting_penalty = 0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] is None:
                wait_duration = self.current_step - data["wait_start"]
                # Progressive penalty that grows more quickly after threshold
                if wait_duration <= 8:
                    waiting_penalty -= 0.1 * wait_duration
                else:
                    waiting_penalty -= 0.1 * 8 + 0.4 * (wait_duration - 8)
        
        # Combine all components
        total_reward = (reward + 
                        movement_penalty + 
                        pickup_reward + 
                        delivery_reward + 
                        positioning_reward + 
                        waiting_penalty)
        
        return total_reward

    def step(self, action):
        # Increment step counter
        self.current_step += 1
        
        # Process actions for each elevator
        for elev_id in range(self.num_elevators):
            elev_action = action[elev_id]
            elev_pos = self.elevator_positions[elev_id]
            
            # Process action: 0 (idle), 1 (up), 2 (down)
            if elev_action == 1 and elev_pos < self.num_floors - 1:  # Up
                self.elevator_positions[elev_id] += 1
                self.elevator_directions[elev_id] = 1
            elif elev_action == 2 and elev_pos > 0:  # Down
                self.elevator_positions[elev_id] -= 1
                self.elevator_directions[elev_id] = 2
            else:  # Idle or invalid move
                self.elevator_directions[elev_id] = 0
                
            # Process current floor (pickup/dropoff passengers)
            self._process_floor(elev_id)
        
        # Generate new passengers with some probability
        if self.np_random.random() < self.passenger_rate:
            self._generate_passenger()
            
        # Increase waiting time for all waiting passengers
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] is None:  # Passenger still waiting
                pass  # Waiting time is tracked by step count difference

        # calculate reward with the function
        reward = self.calculate_reward()

        # Calculate reward (negative total waiting time and passengers in elevators)
        # reward = -(np.sum(self.waiting_passengers) + 
        #          sum(len(passengers) for passengers in self.passengers_in_elevators))
        
        # penalty for movement
        # old reward 0.5*
        #reward -= 0.75 * np.sum(np.array(action) != 0)
        
        # Extra penalty if there is no demand but elevators are moving
        # old reward: reward -= np.sum(np.array(action) != 0
        #if (np.sum(self.waiting_passengers) == 0 and 
        #    sum(len(passengers) for passengers in self.passengers_in_elevators) == 0):
        #    reward -= 3 * np.sum(np.array(action) != 0)

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _process_floor(self, elevator_id):
        """Process passenger pickup and dropoff at current floor for a specific elevator"""
        current_floor = self.elevator_positions[elevator_id]
        
        # Dropoff passengers whose destination is this floor
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
            
        # Turn off elevator button for this floor after dropoff
        if not any(self.waiting_time[p]["destination_floor"] == current_floor 
                  for p in self.passengers_in_elevators[elevator_id]):
            self.elevator_buttons[current_floor, elevator_id] = False
            
        # Pickup waiting passengers if any are waiting at this floor
        if self.waiting_passengers[current_floor] > 0:
            passengers_to_pickup = []
            for passenger_id, data in self.waiting_time.items():
                if (data["start_floor"] == current_floor and 
                    data["wait_end"] is None and
                    data["elevator_id"] is None):  # Not yet picked up by any elevator
                    passengers_to_pickup.append(passenger_id)
            
            # Pick up passengers
            for passenger_id in passengers_to_pickup:
                self.passengers_in_elevators[elevator_id].append(passenger_id)
                self.waiting_time[passenger_id]["wait_end"] = self.current_step
                self.waiting_time[passenger_id]["elevator_id"] = elevator_id
                dest_floor = self.waiting_time[passenger_id]["destination_floor"]
                self.elevator_buttons[dest_floor, elevator_id] = True
                
                # Decrease waiting passenger count
                self.waiting_passengers[current_floor] -= 1
                
            # Check if we picked up all waiting passengers at this floor
            if self.waiting_passengers[current_floor] == 0:
                self.floor_buttons[current_floor] = False
    
    def _render_setup(self):
        """Set up the pygame rendering"""
        pygame.init()
        self.screen_width = 800  # Wider to accommodate multiple elevators
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Multi-Elevator Simulator")
        self.clock = pygame.time.Clock()
        
        # Define colors
        self.colors = {
            "background": (240, 240, 240),
            "elevator": [(100, 100, 100), (120, 80, 80), (80, 120, 80), (80, 80, 120)],  # Different colors for each elevator
            "floor": (200, 200, 200),
            "text": (0, 0, 0),
            "button_off": (150, 150, 150),
            "button_on": (255, 0, 0),
        }
        
        # Load fonts
        self.font = pygame.font.SysFont(None, 24)
    
    def _render_frame(self):
        """Render the current state of the environment"""
        if self.screen is None:
            self._render_setup()
            
        # Clear screen
        self.screen.fill(self.colors["background"])
        
        # Calculate dimensions
        floor_height = self.screen_height // (self.num_floors + 1)
        elevator_width = 60
        elevator_height = floor_height - 10
        elevator_spacing = self.screen_width // (self.num_elevators + 2)
        
        # Draw floors
        for floor in range(self.num_floors):
            # Calculate y position (reversed so floor 0 is at bottom)
            y = self.screen_height - (floor + 1) * floor_height
            
            # Draw floor line
            pygame.draw.line(self.screen, self.colors["floor"], 
                            (0, y + floor_height), 
                            (self.screen_width, y + floor_height), 3)
            
            # Floor number
            text = self.font.render(f"Floor {floor}", True, self.colors["text"])
            self.screen.blit(text, (10, y + floor_height // 2 - 10))
            
            # Floor button (on right side)
            button_color = self.colors["button_on"] if self.floor_buttons[floor] else self.colors["button_off"]
            pygame.draw.circle(self.screen, button_color, 
                              (self.screen_width - 30, y + floor_height // 2), 15)
            
            # Waiting passengers count
            if self.waiting_passengers[floor] > 0:
                text = self.font.render(f"Waiting: {self.waiting_passengers[floor]}", True, self.colors["text"])
                self.screen.blit(text, (self.screen_width - 150, y + floor_height // 2 - 30))
        
        # Draw elevators
        for elev_id in range(self.num_elevators):
            elevator_x = (elev_id + 1) * elevator_spacing - elevator_width // 2
            elevator_y = self.screen_height - (self.elevator_positions[elev_id] + 1) * floor_height
            
            # Use different color for each elevator
            elevator_color = self.colors["elevator"][min(elev_id, len(self.colors["elevator"])-1)]
            pygame.draw.rect(self.screen, elevator_color, 
                            (elevator_x, elevator_y, elevator_width, elevator_height))
            
            # Draw elevator ID
            text = self.font.render(f"E{elev_id}", True, (255, 255, 255))
            text_rect = text.get_rect(center=(elevator_x + elevator_width // 2, 
                                            elevator_y + 20))
            self.screen.blit(text, text_rect)
            
            # Draw elevator buttons
            btn_size = 15
            btn_margin = 3
            btn_start_x = elevator_x + 5
            btn_start_y = elevator_y + 40
            
            for floor in range(self.num_floors):
                btn_row = floor // 5
                btn_col = floor % 5
                btn_x = btn_start_x + btn_col * (btn_size + btn_margin)
                btn_y = btn_start_y + btn_row * (btn_size + btn_margin)
                
                button_color = self.colors["button_on"] if self.elevator_buttons[floor, elev_id] else self.colors["button_off"]
                pygame.draw.rect(self.screen, button_color, 
                                (btn_x, btn_y, btn_size, btn_size))
                
                # Only show floor numbers for first elevator to avoid clutter
                if elev_id == 0:
                    text = self.font.render(str(floor), True, (0, 0, 0))
                    text_rect = text.get_rect(center=(btn_x + btn_size // 2, btn_y + btn_size // 2))
                    self.screen.blit(text, text_rect)
            
            # Draw passengers in elevator
            text = self.font.render(f"P: {len(self.passengers_in_elevators[elev_id])}", True, (255, 255, 255))
            self.screen.blit(text, (elevator_x, elevator_y + elevator_height - 20))
        
        # Draw info
        info_text = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Total served: {self.total_passengers_served}",
            f"Avg wait time: {self.total_waiting_time / max(1, self.total_passengers_served):.2f} steps"
        ]
        
        for i, text in enumerate(info_text):
            surf = self.font.render(text, True, self.colors["text"])
            self.screen.blit(surf, (10, 10 + i * 25))
            
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        pygame.event.pump()
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# Main execution with multiple elevators
if __name__ == "__main__":
    # Initialize the environment with 3 elevators
    env = MultiElevatorEnv(render_mode="human", num_elevators=3)
    observation, info = env.reset()
    
    # Initialize intended directions for each elevator
    # We'll use different patterns for each elevator for better floor coverage
    intended_directions = [1, 2, 1]  # Elevator 0: up, Elevator 1: down, Elevator 2: up
    
    # Initialize episode metrics
    episode_reward = 0
    total_moves = 0
    episode = 1
    
    # Run for multiple episodes
    for _ in range(5000):
        # Determine actions for each elevator
        actions = []
        
        for elev_id in range(env.num_elevators):
            pos = observation["elevator_positions"][elev_id]
            
            # Apply different strategies for each elevator
            if elev_id == 0:
                # First elevator: Simple up/down pattern
                if intended_directions[elev_id] == 1:  # Going up
                    if pos < env.num_floors - 1:
                        action = 1  # Move up
                    else:
                        intended_directions[elev_id] = 2  # Switch to down
                        action = 2  # Move down
                else:  # Going down
                    if pos > 0:
                        action = 2  # Move down
                    else:
                        intended_directions[elev_id] = 1  # Switch to up
                        action = 1  # Move up
            
            elif elev_id == 1:
                # Second elevator: Focus on upper half of building
                if pos < env.num_floors // 2:
                    action = 1  # Move up to upper half
                elif intended_directions[elev_id] == 1:  # Going up in upper half
                    if pos < env.num_floors - 1:
                        action = 1  # Continue up
                    else:
                        intended_directions[elev_id] = 2  # Switch to down
                        action = 2  # Move down
                else:  # Going down in upper half
                    if pos > env.num_floors // 2:
                        action = 2  # Continue down
                    else:
                        intended_directions[elev_id] = 1  # Switch to up
                        action = 1  # Move up
            
            else:
                # Third elevator: Focus on lower half of building
                if pos >= env.num_floors // 2:
                    action = 2  # Move down to lower half
                elif intended_directions[elev_id] == 1:  # Going up in lower half
                    if pos < env.num_floors // 2 - 1:
                        action = 1  # Continue up
                    else:
                        intended_directions[elev_id] = 2  # Switch to down
                        action = 2  # Move down
                else:  # Going down in lower half
                    if pos > 0:
                        action = 2  # Continue down
                    else:
                        intended_directions[elev_id] = 1  # Switch to up
                        action = 1  # Move up
            
            actions.append(action)
        
        # Take the actions
        observation, reward, terminated, truncated, info = env.step(actions)
        
        # Update episode metrics
        episode_reward += reward
        total_moves += sum(1 for a in actions if a != 0)
        
        # Check if episode has finished
        if terminated or truncated:
            avg_wait_time = info.get("average_waiting_time", 0)
            print(f"Episode {episode} finished. Total reward: {episode_reward:.2f}, Total movements: {total_moves}, Average waiting time: {avg_wait_time:.2f} steps")
            episode += 1
            episode_reward = 0
            total_moves = 0
            observation, info = env.reset()
            intended_directions = [1, 2, 1]  # Reset directions
    
    # Close the environment
    env.close()