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
        self.num_floors = 10
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
        
        # Initialize environment state
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.colors = None
        self.font = None
        
        # Reset environment to initialize all state variables
        self.reset()
        
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
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)
        
        return {
            "waiting_passengers": np.sum(self.waiting_passengers),
            "passengers_in_elevators": passengers_in_elevators_count,
            "average_waiting_time": avg_wait_time,
            "total_passengers_served": self.total_passengers_served,
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset environment state
        self.elevator_positions = np.zeros(self.num_elevators, dtype=np.int32)
        self._distribute_elevators()
        
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
        for _ in range(5):
            self._generate_passenger()
        
        # Initialize renderer if needed
        if self.render_mode == "human" and self.screen is None:
            self._render_setup()
            
        return self._get_obs(), self._get_info()
    
    def _distribute_elevators(self):
        """Distribute elevators across floors for better initial coverage"""
        if self.num_elevators > 1:
            # One at bottom, one at top, others evenly spaced
            self.elevator_positions[0] = 0
            self.elevator_positions[-1] = self.num_floors - 1
            for i in range(1, self.num_elevators - 1):
                self.elevator_positions[i] = (i * self.num_floors) // self.num_elevators
    
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
            "elevator_id": None
        }
        
        return passenger_id
    
    def calculate_reward(self):
        """Calculate reward balancing waiting time, pickups, deliveries and movement efficiency"""
        # Base reward
        reward = -0.1
        
        # Movement penalty
        movement_penalty = self._calculate_movement_penalty()
        
        # Pickup and delivery rewards
        pickup_reward = self._calculate_pickup_reward()
        delivery_reward = self._calculate_delivery_reward()
        
        # Strategic positioning reward
        positioning_reward = self._calculate_positioning_reward()
        
        # Waiting time penalty
        waiting_penalty = self._calculate_waiting_penalty()
        
        # Combine all components
        total_reward = (reward + 
                        movement_penalty + 
                        pickup_reward + 
                        delivery_reward + 
                        positioning_reward + 
                        waiting_penalty)
        
        return total_reward
    
    def _calculate_movement_penalty(self):
        """Calculate penalty for elevator movement"""
        movement_penalty = 0
        for elev_id in range(self.num_elevators):
            if self.elevator_directions[elev_id] != 0:
                # Check if elevator has passengers or is moving toward waiting passengers
                has_passengers = len(self.passengers_in_elevators[elev_id]) > 0
                moving_to_pickup = self._is_moving_to_pickup(elev_id)
                
                # Apply penalty with different weights based on purpose
                if has_passengers or moving_to_pickup:
                    movement_penalty -= 0.2  # Reduced penalty for purposeful movement
                else:
                    movement_penalty -= 0.4  # Higher penalty for "speculative" movement
        
        return movement_penalty
    
    def _is_moving_to_pickup(self, elevator_id):
        """Check if elevator is moving toward waiting passengers"""
        curr_pos = self.elevator_positions[elevator_id]
        curr_dir = self.elevator_directions[elevator_id]
        
        if curr_dir == 1:  # Moving up
            for floor in range(curr_pos + 1, self.num_floors):
                if self.waiting_passengers[floor] > 0:
                    return True
        elif curr_dir == 2:  # Moving down
            for floor in range(curr_pos - 1, -1, -1):
                if self.waiting_passengers[floor] > 0:
                    return True
        
        return False
    
    def _calculate_pickup_reward(self):
        """Calculate reward for picking up passengers"""
        pickup_reward = 0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] == self.current_step:
                waiting_duration = data["wait_end"] - data["wait_start"]
                # Exponentially decreasing reward based on wait time
                pickup_reward += 7.0 * (0.9 ** waiting_duration)
        
        return pickup_reward
    
    def _calculate_delivery_reward(self):
        """Calculate reward for delivering passengers"""
        delivery_reward = 0
        for passenger_id, data in self.waiting_time.items():
            if data["travel_end"] == self.current_step:
                delivery_reward += 10.0
                total_trip_time = data["travel_end"] - data["wait_start"]
                # More generous time bonus
                time_bonus = 15.0 * (0.95 ** total_trip_time)
                delivery_reward += time_bonus
        
        return delivery_reward
    
    def _calculate_positioning_reward(self):
        """Calculate reward for strategic positioning of elevators"""
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
        
        return positioning_reward
    
    def _calculate_waiting_penalty(self):
        """Calculate penalty for passenger waiting time"""
        waiting_penalty = 0
        for passenger_id, data in self.waiting_time.items():
            if data["wait_end"] is None:
                wait_duration = self.current_step - data["wait_start"]
                # Progressive penalty that grows more quickly after threshold
                if wait_duration <= 8:
                    waiting_penalty -= 0.1 * wait_duration
                else:
                    waiting_penalty -= 0.1 * 8 + 0.4 * (wait_duration - 8)
        
        return waiting_penalty

    def step(self, action):
        # Increment step counter
        self.current_step += 1
        
        # Process actions for each elevator
        for elev_id in range(self.num_elevators):
            self._process_elevator_action(elev_id, action[elev_id])
        
        # Generate new passengers with some probability
        if self.np_random.random() < self.passenger_rate:
            self._generate_passenger()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Render if needed
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _process_elevator_action(self, elevator_id, action):
        """Process action for a specific elevator"""
        elev_pos = self.elevator_positions[elevator_id]
        
        # Process action: 0 (idle), 1 (up), 2 (down)
        if action == 1 and elev_pos < self.num_floors - 1:  # Up
            self.elevator_positions[elevator_id] += 1
            self.elevator_directions[elevator_id] = 1
        elif action == 2 and elev_pos > 0:  # Down
            self.elevator_positions[elevator_id] -= 1
            self.elevator_directions[elevator_id] = 2
        else:  # Idle or invalid move
            self.elevator_directions[elevator_id] = 0
            
        # Process current floor (pickup/dropoff passengers)
        self._process_floor(elevator_id)
    
    def _process_floor(self, elevator_id):
        """Process passenger pickup and dropoff at current floor for a specific elevator"""
        current_floor = self.elevator_positions[elevator_id]
        
        # Dropoff passengers whose destination is this floor
        self._process_dropoff(elevator_id, current_floor)
            
        # Pickup waiting passengers if any are waiting at this floor
        self._process_pickup(elevator_id, current_floor)
    
    def _process_dropoff(self, elevator_id, current_floor):
        """Process passenger dropoff at current floor"""
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
    
    def _process_pickup(self, elevator_id, current_floor):
        """Process passenger pickup at current floor"""
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
        self.screen_width = 800
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Multi-Elevator Simulator")
        self.clock = pygame.time.Clock()
        
        # Define colors
        self.colors = {
            "background": (240, 240, 240),
            "elevator": [(100, 100, 100), (120, 80, 80), (80, 120, 80), (80, 80, 120)],
            "floor": (200, 200, 200),
            "text": (0, 0, 0),
            "button_off": (150, 150, 150),
            "button_on": (255, 0, 0),
        }
        
        # Load fonts
        self.font = pygame.font.SysFont(None, 24)
        self.button_font = pygame.font.SysFont(None, 12)
    
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
        self._draw_floors(floor_height)
        
        # Draw elevators
        self._draw_elevators(elevator_spacing, elevator_width, elevator_height, floor_height)
        
        # Draw info
        self._draw_info()
            
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        pygame.event.pump()
    
    def _draw_floors(self, floor_height):
        """Draw floor lines, labels and buttons"""
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
    
    def _draw_elevators(self, elevator_spacing, elevator_width, elevator_height, floor_height):
        """Draw elevators with buttons and passengers"""
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
            self._draw_elevator_buttons(elev_id, elevator_x, elevator_y)
            
            # Draw passengers in elevator
            text = self.font.render(f"P: {len(self.passengers_in_elevators[elev_id])}", True, (255, 255, 255))
            self.screen.blit(text, (elevator_x, elevator_y + elevator_height - 20))
    
    def _draw_elevator_buttons(self, elev_id, elevator_x, elevator_y):
        """Draw buttons inside elevator"""
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
            
            text_color = (255, 255, 255) if self.elevator_buttons[floor, elev_id] else (0, 0, 0)
            text = self.button_font.render(str(floor), True, text_color)
            text_rect = text.get_rect(center=(btn_x + btn_size // 2, btn_y + btn_size // 2))
            self.screen.blit(text, text_rect)
    
    def _draw_info(self):
        """Draw simulation information"""
        avg_wait_time = self.total_waiting_time / max(1, self.total_passengers_served)
        info_text = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Total served: {self.total_passengers_served}",
            f"Avg wait time: {avg_wait_time:.2f} steps"
        ]
        
        for i, text in enumerate(info_text):
            surf = self.font.render(text, True, self.colors["text"])
            self.screen.blit(surf, (10, 10 + i * 25))
    
    def close(self):
        """Close the environment and pygame if open"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# # Main execution with multiple elevators
# def run_simulation(num_elevators=3, max_episodes=5, max_steps=500):
#     """Run the elevator simulation with a simple strategy"""
#     env = MultiElevatorEnv(render_mode="human", num_elevators=num_elevators, max_steps=max_steps)
#     observation, info = env.reset()
    
#     # Initialize intended directions for each elevator
#     intended_directions = [1] * num_elevators
#     if num_elevators > 1:
#         intended_directions[1] = 2  # Second elevator starts going down
    
#     # Initialize episode metrics
#     episode_reward = 0
#     total_moves = 0
#     episode = 1
    
#     try:
#         # Run for multiple episodes
#         for _ in range(max_episodes * max_steps):
#             # Determine actions for each elevator
#             actions = []
            
#             for elev_id in range(env.num_elevators):
#                 action = determine_elevator_action(elev_id, observation, intended_directions, env.num_floors)
#                 actions.append(action)
            
#             # Take the actions
#             observation, reward, terminated, truncated, info = env.step(actions)
            
#             # Update episode metrics
#             episode_reward += reward
#             total_moves += sum(1 for a in actions if a != 0)
            
#             # Check if episode has finished
#             if terminated or truncated:
#                 avg_wait_time = info.get("average_waiting_time", 0)
#                 total_served= info.get("total_passengers_served",0)
#                 print(f"Episode {episode} finished. Total reward: {episode_reward:.2f}, Total movements: {total_moves}, Average waiting time: {avg_wait_time:.2f} steps, Total passangers served: {total_served:.2f}")
#                 episode += 1
#                 episode_reward = 0
#                 total_moves = 0
#                 observation, info = env.reset()
#                 intended_directions = [1] * num_elevators
#                 if num_elevators > 1:
#                     intended_directions[1] = 2
                
#                 if episode > max_episodes:
#                     break
    
#     except KeyboardInterrupt:
#         print("Simulation interrupted by user")
#     finally:
#         # Close the environment
#         env.close()


# def determine_elevator_action(elev_id, observation, intended_directions, num_floors):
#     """Determine action for an elevator based on its position and intended direction"""
#     pos = observation["elevator_positions"][elev_id]
    
#     # Apply different strategies for each elevator
#     if elev_id == 0:
#         # First elevator: Simple up/down pattern
#         if intended_directions[elev_id] == 1:  # Going up
#             if pos < num_floors - 1:
#                 action = 1  # Move up
#             else:
#                 intended_directions[elev_id] = 2  # Switch to down
#                 action = 2  # Move down
#         else:  # Going down
#             if pos > 0:
#                 action = 2  # Move down
#             else:
#                 intended_directions[elev_id] = 1  # Switch to up
#                 action = 1  # Move up
    
#     elif elev_id == 1 and len(intended_directions) > 1:
#         # Second elevator: Focus on upper half of building
#         if pos < num_floors // 2:
#             action = 1  # Move up to upper half
#         elif intended_directions[elev_id] == 1:  # Going up in upper half
#             if pos < num_floors - 1:
#                 action = 1  # Continue up
#             else:
#                 intended_directions[elev_id] = 2  # Switch to down
#                 action = 2  # Move down
#         else:  # Going down in upper half
#             if pos > num_floors // 2:
#                 action = 2  # Continue down
#             else:
#                 intended_directions[elev_id] = 1  # Switch to up
#                 action = 1  # Move up
    
#     else:
#         # Third+ elevator: Focus on lower half of building
#         if pos >= num_floors // 2:
#             action = 2  # Move down to lower half
#         elif intended_directions[elev_id] == 1:  # Going up in lower half
#             if pos < num_floors // 2 - 1:
#                 action = 1  # Continue up
#             else:
#                 intended_directions[elev_id] = 2  # Switch to down
#                 action = 2  # Move down
#         else:  # Going down in lower half
#             if pos > 0:
#                 action = 2  # Continue down
#             else:
#                 intended_directions[elev_id] = 1  # Switch to up
#                 action = 1  # Move up
    
#     return action


# if __name__ == "__main__":
#     run_simulation(num_elevators=3, max_episodes=5)