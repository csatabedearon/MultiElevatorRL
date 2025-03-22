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
git clone https://github.com/yourusername/multi_elevator.git
cd multi_elevator
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

## Usage

### Training

To train a new model:
```bash
python -m multi_elevator.training.train
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

## Development

- The project uses Python 3.8 or higher
- Dependencies are managed through `requirements.txt`
- Code follows PEP 8 style guidelines
- Tests should be written for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.
