# Multi-Elevator RL: An Intelligent Elevator Control System

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-red)](https://stable-baselines3.readthedocs.io/)
[![Environment](https://img.shields.io/badge/Environment-Gymnasium-green)](https://gymnasium.farama.org/)

This project presents a sophisticated multi-elevator control system powered by Deep Reinforcement Learning. The core of the project is a custom `Gymnasium` environment that simulates multiple elevators servicing a configurable number of floors. An agent, trained using the Proximal Policy Optimization (PPO) algorithm from `Stable-Baselines3`, learns to efficiently manage elevator movements to minimize passenger waiting times.

The project features a rich, interactive web interface built with Flask and Socket.IO, allowing for real-time visualization, dynamic configuration, and interaction with the simulation.

## Key Features

-   **Advanced RL Agent:** Utilizes Stable-Baselines3 (PPO) to train a robust agent for a complex, multi-agent control problem.
-   **Custom Gymnasium Environment:** A detailed and configurable environment (`MultiElevator-v0`) simulating elevator dynamics, passenger arrivals, and internal/external calls.
-   **Sophisticated Reward Engineering:** A carefully designed reward function that balances multiple objectives: passenger wait time, travel efficiency, and strategic elevator positioning. See our [Reward Experiments Log](./docs/reward_experiments.md) for details on the iterative design process.
-   **Interactive Web UI:** A real-time dashboard to visualize the simulation, monitor key performance metrics, and dynamically adjust parameters.
    -   Live elevator and passenger tracking.
    -   Real-time charts for rewards, passenger counts, and wait times.
    -   On-the-fly configuration of simulation speed, passenger arrival rate, and more.
    -   Ability to load and test different trained models.
-   **TensorBoard Integration:** Comprehensive logging of training metrics for easy monitoring and analysis.
-   **Packaged & Extensible:** The project is structured as an installable Python package, making it easy to set up and extend.


## Project Structure

The repository is organized to separate the core logic, configuration, training scripts, and outputs.

```
├── MultiElevatorRL/            # Main project directory
├── README.md                   # This file
├── requirements.txt            # Core dependencies for the web app
├── setup.py                    # Makes the project an installable package
├── train_model.py              # High-level script to start model training
│
├── configs/
│   └── default_config.yaml     # Default parameters for the environment and training
│
├── docs/
│   └── reward_experiments.md   # Log of reward function designs and results
│
├── logs/                       # Output logs from training runs
│   └── tensorboard/            # Data for TensorBoard visualization
│
├── models/                     # Saved model files
│   ├── best_models/            # Best performing models from evaluation callbacks
│   └── checkpoints/            # Intermediate training checkpoints
│
└── src/
    └── multi_elevator/         # The main Python package
        ├── environment/        # The Gymnasium environment implementation (env.py)
        ├── config/             # Python-based configuration files
        ├── train.py            # The detailed training and evaluation logic
        └── utils/              # Utilities, including the Flask web application
```

## Installation & Setup

### Prerequisites

-   Python 3.8 or higher
-   (Recommended) A CUDA-capable GPU for significantly faster training.
-   (Windows) Visual C++ Redistributable. [Download here](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MultiElevatorRL.git
    cd MultiElevatorRL
    ```

2.  **Create and activate a virtual environment:**
    -   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the project as an editable package:**
    This command uses `setup.py` to install all necessary dependencies, including the development tools.
    ```bash
    pip install -e .
    ```

## Usage

### 1. Training a New Model

To start training a new agent, run the top-level training script. This script uses the parameters defined in `src/multi_elevator/config/training_config.py` but can be easily modified.

```bash
python train_model.py
```

-   Models will be saved in the `models/` directory.
-   Logs will be generated in the `logs/` directory.

To monitor the training progress in real-time, launch TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

Navigate to `http://localhost:6006` in your browser.

### 2. Running the Interactive Web Interface

Launch the Flask web application to see the environment in action.

```bash
python -m multi_elevator.utils.web_app
```

Navigate to **`http://127.0.0.1:5000`** in your browser.

From the web UI, you can:
-   Start/stop/pause the simulation.
-   Adjust parameters like passenger arrival rate and simulation speed.
-   Manually request an elevator by clicking the call button on any floor.
-   Load a pre-trained model (`.zip` file from the `models/` folder) to see how it performs compared to random actions.

### 3. Configuration

The project's behavior can be customized via configuration files:

-   **`configs/default_config.yaml`**: A general-purpose YAML file for high-level settings.
-   **`src/multi_elevator/config/training_config.py`**: The primary configuration for training, defining hyperparameters, environment settings (floors, elevators), and evaluation frequency.

Modify these files to experiment with different building layouts, traffic patterns, or agent architectures.

## Reward Engineering

The effectiveness of the RL agent is highly dependent on the reward function. We have experimented with numerous reward structures to find a balance between minimizing passenger wait times and optimizing elevator movement.

Our journey and the results of various approaches are documented in [**Reward Function Experiments**](./docs/reward_experiments.md). This is a great place to understand the core logic that drives the agent's decision-making process.

## Contributing

Contributions are welcome! If you'd like to help improve the project, please feel free to fork the repository and submit a pull request.

For local development, install the project with the `dev` extras:
```bash
pip install -e .[dev]
```
This will install tools like `pytest`, `black`, `isort`, and `flake8` for testing and code formatting.