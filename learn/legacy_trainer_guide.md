# Legacy DQN Trainer

This directory contains a Deep Q-Network (DQN) implementation using TensorFlow/Keras for training on the Pong environment.

## Overview

The trainer has been adapted from the original Atari Pong implementation to work with:
- **Gymnasium API** (instead of old gym)
- **Custom Pong environment** (`Pong-v0`) with vector observations
- **Modern TensorFlow/Keras** patterns
- **New project structure** with proper package imports

## Key Changes from Original

### 1. Observation Space
- **Original**: Image frames (84x84x4) requiring CNN
- **Updated**: Vector observations (9-dimensional) using MLP
  - Observation includes: ball position, velocity, paddle positions, scores, etc.

### 2. Environment API
- **Original**: `gym` with 4 return values from `step()`
- **Updated**: `gymnasium` with 5 return values `(obs, reward, terminated, truncated, info)`

### 3. Network Architecture
- **Original**: CNN (Conv2D layers) for image processing
- **Updated**: MLP (Dense layers) for vector processing
  - Architecture: Input(9) → Dense(128) → Dense(128) → Dense(64) → Output(3)

### 4. Imports
- **Original**: Relative imports (`from agent_memory import Memory`)
- **Updated**: Package imports (`from .agent_memory import Memory`)

## Files

- `the_agent.py`: DQN agent implementation with MLP network
- `agent_memory.py`: Experience replay buffer
- `environment.py`: Environment wrapper for gymnasium API
- `main.py`: Main training script
- `debug.py`: Debug script for visualization
- `preprocess_frame.py`: Frame preprocessing (kept for compatibility, not used with vector obs)

## Usage

### Training

```bash
python trainer/main.py
```

### Debug Mode

```bash
python trainer/debug.py
```

## Configuration

Edit `main.py` to adjust hyperparameters:

```python
POSSIBLE_ACTIONS = [0, 1, 2]  # 0: stay, 1: up, 2: down
STARTING_MEM_LEN = 50000
MAX_MEM_LEN = 750000
STARTING_EPSILON = 1.0
LEARN_RATE = 0.00025
OBSERVATION_DIM = 9
```

## Dependencies

- `tensorflow>=2.10.0`
- `gymnasium>=0.29.0`
- `numpy>=1.21.0`
- `matplotlib>=3.5.0`
- `opencv-python>=4.5.0` (for preprocess_frame, optional)

## Notes

- The agent uses epsilon-greedy exploration
- Target network updates every 10,000 learning steps
- Weights are saved every 50,000 timesteps
- Training progress is plotted and saved every 100 episodes

