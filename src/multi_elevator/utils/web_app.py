import os
import warnings
import threading
import time
import logging
# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress PyTorch distributed deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='gevent.monkey')

from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from multi_elevator.environment.env import MultiElevatorEnv
import json
from stable_baselines3 import PPO
import torch

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

# Global state and thread safety
state_lock = threading.Lock()
env = None
model = None
simulation_active = False
simulation_paused = False  # Pause flag
simulation_thread = None
simulation_speed = 1.0  # seconds per step
max_simulation_steps = 1000
using_model = False
num_elevators = 3  # Default number of elevators
num_floors = 10    # Default number of floors

def adjust_environment_for_model() -> None:
    """
    Adjust the environment configuration to match the model's expected observation space.
    """
    global num_elevators, num_floors, env, model
    try:
        if model is None:
            return
        expected_shape = model.policy.observation_space.spaces['elevator_buttons'].shape  # e.g. (5, 3)
        expected_floors, expected_elevators = expected_shape
        current_shape = env.observation_space.spaces['elevator_buttons'].shape
        if current_shape != expected_shape:
            logger.error(
                "Observation space mismatch: current elevator_buttons shape is %s, but model expects %s. Adjusting environment configuration.",
                current_shape, expected_shape
            )
            num_floors = expected_floors
            num_elevators = expected_elevators
            logger.info(
                "Updating configuration to %d floors and %d elevators as required by the model",
                num_floors, num_elevators
            )
            init_environment()
    except Exception as e:
        logger.exception("Error in adjusting environment for model: %s", e)

def init_environment() -> None:
    global env
    with state_lock:
        logger.debug("Initializing environment with %d elevators and %d floors", num_elevators, num_floors)
        env = MultiElevatorEnv(render_mode=None, max_steps=max_simulation_steps,
                               passenger_rate=0.1, num_elevators=num_elevators, num_floors=num_floors)
        env.reset()

@app.route('/update_configuration', methods=['POST'])
def update_configuration():
    """
    Update environment configuration.
    If a model is loaded, check that the new configuration matches the model's expected observation space.
    The client can send an extra key 'override' (boolean) to force the change.
    """
    global num_elevators, num_floors, simulation_active, env, model, using_model
    try:
        logger.debug("Received configuration update request")
        data = request.json
        if not data:
            logger.error("No JSON data provided")
            return jsonify({'status': 'error', 'message': 'No configuration data provided'})
        
        new_elevators = int(data.get('num_elevators', 3))
        new_floors = int(data.get('num_floors', 10))
        override = data.get('override', False)
        
        # If using a model, check the expected observation shape
        if using_model and model is not None:
            expected_shape = model.policy.observation_space.spaces['elevator_buttons'].shape  # e.g. (5, 3)
            expected_floors, expected_elevators = expected_shape
            if (new_floors, new_elevators) != (expected_floors, expected_elevators) and not override:
                msg = (
                    f"Model is trained on {expected_floors} floors and {expected_elevators} elevators. "
                    "To force configuration change, send override=true."
                )
                logger.error(msg)
                return jsonify({
                    'status': 'error',
                    'message': msg,
                    'expected_configuration': {
                        'num_floors': expected_floors,
                        'num_elevators': expected_elevators
                    }
                })
            elif override:
                logger.info("Override flag received. Proceeding with new configuration (%d floors, %d elevators).", new_floors, new_elevators)
        
        # Validate input if no model is forcing a configuration
        if new_elevators < 1 or new_elevators > 10:
            logger.error("Invalid elevator count: %d", new_elevators)
            return jsonify({'status': 'error', 'message': 'Number of elevators must be between 1 and 10'})
        if new_floors < 2 or new_floors > 20:
            logger.error("Invalid floor count: %d", new_floors)
            return jsonify({'status': 'error', 'message': 'Number of floors must be between 2 and 20'})
            
        num_elevators = new_elevators
        num_floors = new_floors
        logger.debug("Configuration updated: Elevators=%d, Floors=%d", num_elevators, num_floors)
        
        if simulation_active:
            logger.debug("Stopping active simulation for reconfiguration")
            simulation_active = False
            if simulation_thread:
                simulation_thread.join(timeout=2)
        
        logger.debug("Reinitializing environment with new configuration")
        init_environment()
        logger.debug("Environment reinitialized successfully")
        
        return jsonify({'status': 'success', 'num_elevators': num_elevators, 'num_floors': num_floors})
    except ValueError as ve:
        logger.exception("Value error in configuration update")
        return jsonify({'status': 'error', 'message': f"Value error in configuration: {str(ve)}"})
    except Exception as e:
        logger.exception("Unexpected error in configuration update")
        return jsonify({'status': 'error', 'message': f"Unexpected error in configuration: {str(e)}"})

@app.route('/update_passenger_rate', methods=['POST'])
def update_passenger_rate():
    global env
    try:
        new_rate = float(request.json['rate'])
        with state_lock:
            if env:
                env.passenger_rate = new_rate
        return jsonify({'status': 'success', 'new_rate': new_rate})
    except Exception as e:
        logger.exception("Error updating passenger rate")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_simulation_speed', methods=['POST'])
def update_simulation_speed():
    global simulation_speed
    try:
        new_speed = float(request.json['speed'])
        simulation_speed = max(0.1, min(2.0, new_speed))
        return jsonify({'status': 'success', 'new_speed': simulation_speed})
    except Exception as e:
        logger.exception("Error updating simulation speed")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_max_steps', methods=['POST'])
def update_max_steps():
    global max_simulation_steps, env
    try:
        new_max_steps = int(request.json['max_steps'])
        max_simulation_steps = max(100, min(5000, new_max_steps))
        with state_lock:
            if env:
                env.max_steps = max_simulation_steps
        return jsonify({'status': 'success', 'new_max_steps': max_simulation_steps})
    except Exception as e:
        logger.exception("Error updating max steps")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/pause_simulation', methods=['POST'])
def pause_simulation():
    global simulation_paused
    try:
        data = request.json
        should_pause = data.get('pause', True)
        simulation_paused = should_pause
        return jsonify({
            'status': 'success',
            'paused': simulation_paused
        })
    except Exception as e:
        logger.exception("Error toggling pause")
        return jsonify({'status': 'error', 'message': str(e)})

def simulation_thread_function() -> None:
    global env, simulation_active, simulation_paused, simulation_speed, max_simulation_steps, using_model, model
    
    # Initialize episode statistics
    episode_stats = {
        'total_reward': 0,
        'total_passengers_served': 0,
        'total_waiting_time': 0,
        'max_waiting_time': 0,
        'min_waiting_time': float('inf'),
        'steps_taken': 0,
        'avg_passengers_per_step': 0,
        'total_passengers_generated': 0
    }
    
    while simulation_active and env.current_step < max_simulation_steps:
        if simulation_paused:
            time.sleep(0.1)
            continue

        with state_lock:
            if using_model and model is not None:
                try:
                    obs = env._get_obs()
                    try:
                        action, _ = model.predict(obs)
                    except ValueError as e:
                        if "Unexpected observation shape" in str(e):
                            logger.error("Observation shape mismatch during predict: %s", e)
                            adjust_environment_for_model()
                            continue
                        else:
                            raise
                except Exception as ex:
                    logger.exception("Error during model prediction: %s", ex)
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

        # Update episode statistics
        episode_stats['total_reward'] += reward
        episode_stats['steps_taken'] = env.current_step + 1
        episode_stats['total_passengers_generated'] = len(env.waiting_time)

        # Calculate waiting time statistics
        total_wait_time = 0
        completed_passengers = 0
        for data in env.waiting_time.values():
            if data["travel_end"] is not None:
                wait_time = data["wait_end"] - data["wait_start"]
                total_wait_time += wait_time
                completed_passengers += 1
                episode_stats['max_waiting_time'] = max(episode_stats['max_waiting_time'], wait_time)
                episode_stats['min_waiting_time'] = min(episode_stats['min_waiting_time'], wait_time)

        episode_stats['total_passengers_served'] = completed_passengers
        episode_stats['total_waiting_time'] = total_wait_time
        avg_wait_time = total_wait_time / completed_passengers if completed_passengers > 0 else 0
        episode_stats['avg_passengers_per_step'] = len(env.waiting_time) / (env.current_step + 1)

        socketio.emit('stats_update', {
            'elevator_positions': env.elevator_positions.tolist(),
            'elevator_directions': env.elevator_directions.tolist(),
            'waiting_passengers': env.waiting_passengers.tolist(),
            'elevator_passengers': [len(passengers) for passengers in env.passengers_in_elevators],
            'reward': reward,
            'current_step': env.current_step,
            'max_steps': max_simulation_steps,
            'passenger_rate': env.passenger_rate,
            'simulation_speed': simulation_speed,
            'using_model': using_model,
            'avg_wait_time': avg_wait_time
        })

        if terminated or truncated or env.current_step == max_simulation_steps - 1:
            # Calculate final statistics
            if episode_stats['min_waiting_time'] == float('inf'):
                episode_stats['min_waiting_time'] = 0
                
            avg_waiting_time = (episode_stats['total_waiting_time'] / 
                              episode_stats['total_passengers_served'] if episode_stats['total_passengers_served'] > 0 else 0)
            
            # Emit episode summary
            socketio.emit('episode_summary', {
                'total_reward': round(episode_stats['total_reward'], 2),
                'avg_reward': round(episode_stats['total_reward'] / episode_stats['steps_taken'], 2),
                'total_passengers_served': episode_stats['total_passengers_served'],
                'total_passengers_generated': episode_stats['total_passengers_generated'],
                'completion_rate': round(episode_stats['total_passengers_served'] / 
                                      max(episode_stats['total_passengers_generated'], 1) * 100, 1),
                'avg_waiting_time': round(avg_waiting_time, 2),
                'min_waiting_time': round(episode_stats['min_waiting_time'], 2),
                'max_waiting_time': round(episode_stats['max_waiting_time'], 2),
                'total_steps': episode_stats['steps_taken'],
                'avg_passengers_per_step': round(episode_stats['avg_passengers_per_step'], 2),
                'using_model': using_model
            })
            
            simulation_active = False
            break

        time.sleep(simulation_speed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    global simulation_active, simulation_thread
    try:
        if not simulation_active:
            simulation_active = True
            init_environment()
            simulation_thread = threading.Thread(target=simulation_thread_function, daemon=True)
            simulation_thread.start()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("Error starting simulation")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    global simulation_active, simulation_thread
    simulation_active = False
    if simulation_thread:
        simulation_thread.join(timeout=2)
    return jsonify({'status': 'success'})

@app.route('/load_model', methods=['POST'])
def load_model():
    global model, using_model, num_floors, num_elevators, env
    if 'model_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'})
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    try:
        # Stop any running simulation
        global simulation_active, simulation_thread
        simulation_active = False
        if simulation_thread:
            simulation_thread.join(timeout=2)
        
        temp_path = 'temp_model.zip'
        file.save(temp_path)
        
        # Load the model
        model = PPO.load(temp_path)
        using_model = True
        
        # Extract expected configuration from the model's observation space
        expected_buttons_shape = model.policy.observation_space.spaces['elevator_buttons'].shape
        expected_num_floors, expected_num_elevators = expected_buttons_shape
        
        # Force the global configuration to match the model
        num_floors = expected_num_floors
        num_elevators = expected_num_elevators
        
        # Reinitialize environment with the model's required configuration
        init_environment()
        
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Model loaded successfully. Environment configured for {num_floors} floors and {num_elevators} elevators.',
            'num_floors': num_floors,
            'num_elevators': num_elevators
        })
    except Exception as e:
        using_model = False
        model = None
        logger.exception("Error loading model")
        return jsonify({'status': 'error', 'message': f'Error loading model: {str(e)}'})

@app.route('/request_elevator', methods=['POST'])
def request_elevator():
    global env
    try:
        data = request.json
        floor = int(data.get('floor', 0))
        if env is None:
            return jsonify({'status': 'error', 'message': 'Environment not initialized'})
        if floor < 0 or floor >= env.num_floors:
            return jsonify({'status': 'error', 'message': 'Invalid floor number'})
        with state_lock:
            env._generate_passenger(start_floor=floor)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("Error requesting elevator")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/reset_simulation', methods=['POST'])
def reset_simulation():
    global env, simulation_active, simulation_thread, model, using_model
    try:
        simulation_active = False
        if simulation_thread:
            simulation_thread.join(timeout=2)
        init_environment()
        model = None
        using_model = False
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.exception("Error resetting simulation")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 