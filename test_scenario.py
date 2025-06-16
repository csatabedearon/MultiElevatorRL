# test_scenario.py
from src.multi_elevator.environment.env import MultiElevatorEnv
import numpy as np

# 1. Define a simple scenario
my_test_scenario = [
    {"arrival_step": 5, "start_floor": 0, "destination_floor": 4},
    {"arrival_step": 10, "start_floor": 3, "destination_floor": 0},
    {"arrival_step": 10, "start_floor": 2, "destination_floor": 1},
]

# 2. Initialize the environment with the scenario
# Using 2 elevators and 5 floors for this test
env = MultiElevatorEnv(
    num_elevators=2,
    num_floors=5,
    scenario=my_test_scenario
)

obs, info = env.reset()

print("--- Starting Scenario Test ---")
print(f"Initial Waiting Passengers: {obs['waiting_passengers']}") # Should be all zeros

# 3. Run the simulation for a few steps
for i in range(15):
    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n--- Step {env.current_step} ---")
    print(f"Waiting Passengers: {obs['waiting_passengers']}")
    
    # Check if passengers were spawned correctly
    if env.current_step == 5:
        print(">> Check: A passenger should have appeared at floor 0.")
    if env.current_step == 10:
        print(">> Check: Passengers should have appeared at floors 2 and 3.")

env.close()
print("\n--- Test Complete ---")