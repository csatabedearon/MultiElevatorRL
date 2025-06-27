"""
This module implements a Flask-based web application for visualizing and interacting
with the Multi-Elevator environment.

It uses Flask and SocketIO to provide a real-time web interface where users can:
- Start, stop, pause, and reset the simulation.
- Adjust simulation parameters like speed, passenger rate, and total steps.
- Load a pre-trained Stable Baselines3 PPO model to control the elevators.
- Manually request an elevator from any floor.
- Load a predefined passenger arrival scenario.
- View real-time statistics and a summary of each episode.

The application runs a separate thread for the simulation loop to keep the UI responsive.
"""
import os
import warnings
import threading
import time
import logging

# Suppress TensorFlow oneDNN warnings for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress potential deprecation warnings from gevent
warnings.filterwarnings('ignore', category=FutureWarning, module='gevent.monkey')

# Patch standard libraries for gevent compatibility
from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

# Import the environment package to ensure the custom Gym environment is registered
import multi_elevator.environment
from multi_elevator.environment.env import MultiElevatorEnv

import json
from stable_baselines3 import PPO
import torch

# Configure logging for the web application
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Flask and SocketIO Setup ---
app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

# --- Global State and Thread Safety ---
state_lock = threading.Lock()  # Lock to ensure thread-safe access to shared state
env: MultiElevatorEnv = None    # The environment instance
model: PPO = None             # The loaded PPO model instance
simulation_active = False     # Flag to control the main simulation loop
simulation_paused = False     # Flag to pause/resume the simulation
simulation_thread = None      # The thread running the simulation loop
simulation_speed = 1.0        # Time in seconds between simulation steps
max_simulation_steps = 1000   # Maximum steps for a simulation episode
using_model = False           # Flag indicating if a trained model is in use
num_elevators = 3             # Default number of elevators
num_floors = 10               # Default number of floors

# --- Scenario-Specific State ---
passenger_scenario = None     # Stores the loaded passenger arrival scenario
scenario_mode_active = False  # Flag indicating if a scenario is currently loaded

def adjust_environment_for_model() -> None:
    """
    Adjusts the environment configuration to match the loaded model's expectations.

    When a pre-trained model is loaded, the environment's number of floors and
    elevators must match the observation space the model was trained on. This function
    ensures that consistency.
    """
    global num_elevators, num_floors, env, model
    if model is None:
        return
    try:
        # Extract the expected shape from the model's observation space
        expected_shape = model.policy.observation_space.spaces['elevator_buttons'].shape
        expected_floors, expected_elevators = expected_shape

        if (num_floors, num_elevators) != (expected_floors, expected_elevators):
            logger.info(
                f"Model requires {expected_floors} floors and {expected_elevators} elevators. "
                f"Adjusting environment from {num_floors} floors and {num_elevators} elevators."
            )
            num_floors = expected_floors
            num_elevators = expected_elevators
            init_environment()  # Re-initialize the environment with the new settings
    except Exception as e:
        logger.exception(f"Error adjusting environment for model: {e}")

def init_environment() -> None:
    """
    Initializes or re-initializes the simulation environment.

    This function creates a new instance of `MultiElevatorEnv` with the current
    global configuration (e.g., number of elevators, floors, and scenario).
    It is thread-safe.
    """
    global env, passenger_scenario, scenario_mode_active
    with state_lock:
        logger.debug(
            f"Initializing environment: {num_elevators} elevators, {num_floors} floors, Scenario Mode: {scenario_mode_active}"
        )
        # Pass the scenario to the environment only if scenario mode is active
        current_scenario = passenger_scenario if scenario_mode_active else None
        env = MultiElevatorEnv(
            render_mode=None,  # Rendering is handled by the web interface
            max_steps=max_simulation_steps,
            passenger_rate=0.1,  # Ignored if a scenario is active
            num_elevators=num_elevators,
            num_floors=num_floors,
            scenario=current_scenario
        )
        env.reset()

@app.route('/update_configuration', methods=['POST'])
def update_configuration():
    """
    Handles requests to update the environment's core configuration.

    Updates the number of elevators and floors. If a model is active, it validates
    the new configuration against the model's requirements unless an override
    flag is provided.
    """
    global num_elevators, num_floors, simulation_active, env, model, using_model
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No configuration data provided'}), 400

        new_elevators = int(data.get('num_elevators', num_elevators))
        new_floors = int(data.get('num_floors', num_floors))
        override = data.get('override', False)

        # If a model is loaded, ensure the new configuration is compatible
        if using_model and model and not override:
            expected_shape = model.policy.observation_space.spaces['elevator_buttons'].shape
            expected_floors, expected_elevators = expected_shape
            if (new_floors, new_elevators) != (expected_floors, expected_elevators):
                msg = f"Model requires {expected_floors} floors and {expected_elevators} elevators."
                return jsonify({
                    'status': 'error',
                    'message': msg,
                    'expected_configuration': {'num_floors': expected_floors, 'num_elevators': expected_elevators}
                }), 409  # 409 Conflict

        # Stop any active simulation before changing the configuration
        if simulation_active:
            stop_simulation()

        num_elevators = new_elevators
        num_floors = new_floors
        init_environment()  # Re-initialize with the new settings

        logger.info(f"Configuration updated: {num_elevators} elevators, {num_floors} floors.")
        return jsonify({'status': 'success', 'num_elevators': num_elevators, 'num_floors': num_floors})

    except (ValueError, TypeError) as e:
        return jsonify({'status': 'error', 'message': f"Invalid data format: {e}"}), 400
    except Exception as e:
        logger.exception("Error updating configuration")
        return jsonify({'status': 'error', 'message': 'An internal error occurred'}), 500

@app.route('/update_passenger_rate', methods=['POST'])
def update_passenger_rate():
    """Updates the passenger arrival rate for random passenger generation."""
    global env
    try:
        new_rate = float(request.json['rate'])
        if not (0.0 <= new_rate <= 1.0):
            return jsonify({'status': 'error', 'message': 'Rate must be between 0.0 and 1.0'}), 400
        with state_lock:
            if env:
                env.passenger_rate = new_rate
        return jsonify({'status': 'success', 'new_rate': new_rate})
    except Exception as e:
        logger.exception("Error updating passenger rate")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_simulation_speed', methods=['POST'])
def update_simulation_speed():
    """Updates the time delay between simulation steps."""
    global simulation_speed
    try:
        new_speed = float(request.json['speed'])
        simulation_speed = max(0.1, min(2.0, new_speed))  # Clamp speed to a reasonable range
        return jsonify({'status': 'success', 'new_speed': simulation_speed})
    except Exception as e:
        logger.exception("Error updating simulation speed")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_max_steps', methods=['POST'])
def update_max_steps():
    """Updates the maximum number of steps for an episode."""
    global max_simulation_steps, env
    try:
        new_max_steps = int(request.json['max_steps'])
        max_simulation_steps = max(100, min(5000, new_max_steps)) # Clamp to a reasonable range
        with state_lock:
            if env:
                env.max_steps = max_simulation_steps
        return jsonify({'status': 'success', 'new_max_steps': max_simulation_steps})
    except Exception as e:
        logger.exception("Error updating max steps")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/pause_simulation', methods=['POST'])
def pause_simulation():
    """Toggles the paused state of the simulation."""
    global simulation_paused
    try:
        should_pause = request.json.get('pause', True)
        simulation_paused = bool(should_pause)
        logger.info(f"Simulation {'paused' if simulation_paused else 'resumed'}.")
        return jsonify({'status': 'success', 'paused': simulation_paused})
    except Exception as e:
        logger.exception("Error toggling pause")
        return jsonify({'status': 'error', 'message': str(e)})

def simulation_thread_function() -> None:
    """
    The main loop for running the environment simulation.

    This function runs in a separate thread. It continuously steps through the
    environment, taking actions (either random or from a model), and broadcasting
    the updated state and statistics to the web client via SocketIO.

    The loop terminates when the simulation is stopped, or the episode ends.
    """
    global env, simulation_active, simulation_paused, simulation_speed, max_simulation_steps, using_model, model

    logger.info("Simulation thread started.")

    while simulation_active and env.current_step < max_simulation_steps:
        if simulation_paused:
            time.sleep(0.1)  # Sleep briefly when paused to avoid busy-waiting
            continue

        with state_lock:
            # Decide on the action: either from the model or random sampling
            if using_model and model:
                try:
                    obs = env._get_obs()
                    action, _ = model.predict(obs, deterministic=True)
                except Exception as e:
                    logger.exception(f"Error during model prediction: {e}. Falling back to random action.")
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            # Execute the action in the environment
            observation, reward, terminated, truncated, info = env.step(action)

        # Broadcast the updated state to all connected clients
        socketio.emit('stats_update', {
            'elevator_positions': env.elevator_positions.tolist(),
            'elevator_directions': env.elevator_directions.tolist(),
            'waiting_passengers': env.waiting_passengers.tolist(),
            'elevator_passengers': [len(p) for p in env.passengers_in_elevators],
            'reward': reward,
            'current_step': env.current_step,
            'max_steps': max_simulation_steps,
            'passenger_rate': env.passenger_rate,
            'simulation_speed': simulation_speed,
            'using_model': using_model,
            'avg_wait_time': info.get('average_waiting_time', 0),
            'scenario_mode': scenario_mode_active
        })

        # If the episode is over, send a final summary and stop the loop
        if terminated or truncated:
            logger.info(f"Episode finished after {env.current_step} steps.")
            socketio.emit('episode_summary', {
                'total_reward': info.get('total_reward', 0),
                'total_passengers_served': info.get('total_passengers_served', 0),
                'avg_waiting_time': info.get('average_waiting_time', 0),
                'total_steps': env.current_step
            })
            simulation_active = False
            break

        time.sleep(simulation_speed)  # Control the speed of the simulation

    logger.info("Simulation thread finished.")

@app.route('/')
def index():
    """Serves the main HTML page of the web application."""
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    """Starts the simulation thread if it is not already active."""
    global simulation_active, simulation_thread
    if not simulation_active:
        simulation_active = True
        init_environment()  # Ensure environment is fresh at the start
        simulation_thread = threading.Thread(target=simulation_thread_function, daemon=True)
        simulation_thread.start()
        logger.info("Simulation started by user.")
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Simulation already running'}), 409

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stops the currently active simulation thread."""
    global simulation_active, simulation_thread
    simulation_active = False
    if simulation_thread and simulation_thread.is_alive():
        simulation_thread.join(timeout=2)  # Wait for the thread to finish
    logger.info("Simulation stopped by user.")
    return jsonify({'status': 'success'})

@app.route('/load_model', methods=['POST'])
def load_model():
    """
    Handles the uploading of a pre-trained PPO model file.

    It stops any running simulation, saves the uploaded file temporarily, loads it
    using Stable Baselines3, and then adjusts the environment to match the model's
    configuration.
    """
    global model, using_model
    if 'model_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'}), 400
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400

    try:
        stop_simulation()  # Ensure no simulation is running

        temp_path = 'temp_model.zip'
        file.save(temp_path)

        model = PPO.load(temp_path)
        using_model = True
        logger.info(f"Successfully loaded model from {file.filename}")

        # Adjust environment to match the loaded model
        adjust_environment_for_model()

        os.remove(temp_path)

        return jsonify({
            'status': 'success',
            'message': f'Model loaded. Environment set to {num_floors} floors and {num_elevators} elevators.',
            'num_floors': num_floors,
            'num_elevators': num_elevators
        })
    except Exception as e:
        using_model = False
        model = None
        logger.exception(f"Error loading model: {e}")
        return jsonify({'status': 'error', 'message': f'Error loading model: {e}'}), 500

@app.route('/load_scenario', methods=['POST'])
def load_scenario():
    """
    Receives and validates a passenger arrival scenario from the client.

    The scenario, a list of passenger arrival events, is validated for correct
    formatting and floor numbers before being activated for the next simulation.
    """
    global passenger_scenario, scenario_mode_active
    stop_simulation()

    data = request.json
    if not isinstance(data, list):
        return jsonify({'status': 'error', 'message': 'Scenario must be a list.'}), 400

    # Validate the structure and values of the scenario data
    for item in data:
        if not all(k in item for k in ['arrival_step', 'start_floor', 'destination_floor']):
            return jsonify({'status': 'error', 'message': 'Scenario item is malformed.'}), 400
        if not (0 <= item['start_floor'] < num_floors and 0 <= item['destination_floor'] < num_floors):
            return jsonify({'status': 'error', 'message': f"Invalid floor number in scenario."}), 400

    with state_lock:
        passenger_scenario = data
        scenario_mode_active = True
        logger.info(f"Loaded scenario with {len(passenger_scenario)} passengers.")

    return jsonify({'status': 'success', 'message': f'Scenario with {len(data)} passengers loaded.'})

@app.route('/request_elevator', methods=['POST'])
def request_elevator():
    """Allows a user to manually generate a new passenger at a specific floor."""
    global env
    try:
        floor = int(request.json.get('floor', 0))
        if not env or not (0 <= floor < env.num_floors):
            return jsonify({'status': 'error', 'message': 'Invalid floor number'}), 400

        with state_lock:
            env._generate_passenger(start_floor=floor)
        logger.info(f"Manual passenger request at floor {floor}.")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("Error processing elevator request")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/reset_simulation', methods=['POST'])
def reset_simulation():
    """
    Resets the entire simulation state to its default configuration.

    This stops any active simulation, unloads any model or scenario, and resets
    the environment to its initial default state.
    """
    global model, using_model, passenger_scenario, scenario_mode_active
    stop_simulation()

    with state_lock:
        model = None
        using_model = False
        passenger_scenario = None
        scenario_mode_active = False
        init_environment()  # Reset to default environment

    logger.info("Simulation reset to default state.")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    """Starts the Flask-SocketIO web server."""
    init_environment()  # Initialize the environment when the app starts
    logger.info("Starting Flask-SocketIO server at http://0.0.0.0:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)