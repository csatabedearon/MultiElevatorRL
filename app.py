import os
import warnings
# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress PyTorch distributed deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='gevent.monkey')

from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from multi_elevator_env import MultiElevatorEnv
import threading
import time
import json
from stable_baselines3 import PPO
import torch

app = Flask(__name__)
socketio = SocketIO(app, async_mode='gevent')

# Global variables
env = None
model = None
simulation_active = False
simulation_thread = None
simulation_speed = 1.0  # seconds between steps
max_simulation_steps = 1000
using_model = False

def init_environment():
    global env
    env = MultiElevatorEnv(render_mode=None, max_steps=max_simulation_steps, passenger_rate=0.1, num_elevators=3)
    env.reset()

@app.route('/update_passenger_rate', methods=['POST'])
def update_passenger_rate():
    global env
    try:
        new_rate = float(request.json['rate'])
        if env:
            env.passenger_rate = new_rate
        return jsonify({'status': 'success', 'new_rate': new_rate})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_simulation_speed', methods=['POST'])
def update_simulation_speed():
    global simulation_speed
    try:
        new_speed = float(request.json['speed'])
        simulation_speed = max(0.1, min(2.0, new_speed))  # Limit between 0.1 and 2 seconds
        return jsonify({'status': 'success', 'new_speed': simulation_speed})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/update_max_steps', methods=['POST'])
def update_max_steps():
    global max_simulation_steps
    try:
        new_max_steps = int(request.json['max_steps'])
        max_simulation_steps = max(100, min(5000, new_max_steps))  # Limit between 100 and 5000 steps
        if env:
            env.max_steps = max_simulation_steps
        return jsonify({'status': 'success', 'new_max_steps': max_simulation_steps})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def simulation_thread_function():
    global env, simulation_active, simulation_speed, max_simulation_steps, using_model
    
    while simulation_active and env.current_step < max_simulation_steps:
        if using_model and model is not None:
            obs = env._get_obs()
            action, _ = model.predict(obs)
            observation, reward, terminated, truncated, info = env.step(action)
        else:
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        # Calculate average waiting time
        total_wait_time = 0
        completed_passengers = 0
        for data in env.waiting_time.values():
            if data["travel_end"] is not None:  # Passenger completed their journey
                wait_time = data["wait_end"] - data["wait_start"]  # Time from appearance to pickup
                total_wait_time += wait_time
                completed_passengers += 1
        
        avg_wait_time = total_wait_time / completed_passengers if completed_passengers > 0 else 0
        
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
            env.reset()
        
        time.sleep(simulation_speed)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    global simulation_active, simulation_thread
    
    if not simulation_active:
        simulation_active = True
        init_environment()
        simulation_thread = threading.Thread(target=simulation_thread_function, daemon=True)
        simulation_thread.start()
    
    return jsonify({'status': 'success'})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    global simulation_active, simulation_thread, model, using_model
    simulation_active = False
    if simulation_thread:
        simulation_thread.join(timeout=2)
    model = None
    using_model = False
    return jsonify({'status': 'success'})

@app.route('/load_model', methods=['POST'])
def load_model():
    global model, using_model
    
    if 'model_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file provided'})
        
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
        
    try:
        # Save the uploaded file temporarily
        temp_path = 'temp_model.zip'
        file.save(temp_path)
        
        # Load the model
        model = PPO.load(temp_path)
        using_model = True
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
    except Exception as e:
        using_model = False
        return jsonify({'status': 'error', 'message': f'Error loading model: {str(e)}'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 