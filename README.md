# Multi-Elevator System with Reinforcement Learning

A sophisticated multi-elevator system that uses reinforcement learning to optimize elevator scheduling and passenger transportation.

## Project Structure

```
multi_elevator/
├── src/
│   └── multi_elevator/
│       ├── environment/     # Environment implementation
│       ├── models/         # Model definitions and architectures
│       ├── training/       # Training scripts and utilities
│       └── utils/          # Utility functions and web interface
├── tests/                 # Test files
├── configs/              # Configuration files
├── models/
│   ├── best_models/      # Best performing model checkpoints
│   └── checkpoints/      # Training checkpoints
├── docs/                 # Documentation
├── logs/                 # Training logs and tensorboard data
├── requirements.txt      # Project dependencies
└── setup.py             # Package installation configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/csatabedearon/MultiElevatorRL.git
cd MultiElevatorRL
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Environment Setup

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- At least 8GB RAM
- 20GB free disk space
- Visual C++ Redistributable (download from: https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Dependencies
The project uses several key libraries:
- PyTorch for deep learning
- Gymnasium for RL environment
- Stable-Baselines3 for RL algorithms
- Flask for web interface
- NumPy and Pandas for data handling
- Matplotlib for visualization

All dependencies are listed in `requirements.txt` and will be installed automatically during setup.

### Configuration
The project uses configuration files in the `configs/` directory:
- `env_config.yaml`: Environment parameters (number of elevators, floors, etc.)
- `model_config.yaml`: Model architecture and hyperparameters
- `training_config.yaml`: Training settings and optimization parameters

You can modify these files to customize the system behavior.

## Usage

### Training

To train a new model:
```bash
python -m multi_elevator.training.train
```

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir ppo_multi_elevator_tensorboard
```

### Testing

To test a trained model:
```bash
python -m multi_elevator.training.test
```

### Web Interface

To run the web interface:
```bash
python -m multi_elevator.utils.web_app
```

The web interface will be available at `http://localhost:5000`

## Development

- The project uses Python 3.8 or higher
- Dependencies are managed through `requirements.txt`
- Code follows PEP 8 style guidelines
- Tests should be written for new features
- Use `pytest` to run the test suite

## License

This project is licensed under the MIT License - see the LICENSE file for details.
