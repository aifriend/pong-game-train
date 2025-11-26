# Pong Gym Environment for Reinforcement Learning

A custom OpenAI Gymnasium environment based on the classic Pong game, designed for training reinforcement learning agents.

## Overview

This environment adapts the traditional Pong game to conform to the [Gymnasium API](https://gymnasium.farama.org/), enabling seamless training with popular RL algorithms and libraries like Stable-Baselines3, Ray RLLib, and more.

## Features

- **Gymnasium-compliant**: Full compatibility with the Gymnasium API
- **Configurable**: Adjustable game parameters (ball speed, max score, episode length)  
- **Multiple render modes**: Human-readable display and RGB array for analysis
- **Reward shaping**: Designed reward system to encourage learning
- **Easy integration**: Works with popular RL libraries out of the box

## Installation

### Prerequisites

Make sure you have Python 3.8+ installed, then create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install the Package

```bash
# Clone the repository
git clone <your-repo-url>
cd Pong

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from pong import register_pong_env

# Register the environment
register_pong_env()

# Create the environment
env = gym.make('Pong-v0', render_mode='human')

# Reset and start
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training with Stable-Baselines3

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pong import register_pong_env

# Register environment
register_pong_env()

# Create vectorized environment for training
env = make_vec_env('Pong-v0', n_envs=4)

# Create and train the agent
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./pong_tensorboard/")
model.learn(total_timesteps=100000)

# Save the model
model.save("pong_ppo")

# Test the trained agent
env = gym.make('Pong-v0', render_mode='human')
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

## Environment Details

### Action Space

The agent has 3 discrete actions:

| Action | Description |
|--------|-------------|
| 0      | Stay/No movement |
| 1      | Move paddle up |
| 2      | Move paddle down |

### Observation Space

The environment returns a 9-dimensional observation vector:

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Ball X position | [0, 960] |
| 1 | Ball Y position | [0, 720] |
| 2 | Ball X velocity | [-4, 4] |
| 3 | Ball Y velocity | [-4, 4] |
| 4 | Player paddle Y position | [20, 700] |
| 5 | Opponent paddle Y position | [20, 700] |
| 6 | Distance from ball to player | [0, ~1200] |
| 7 | Player score | [0, max_score] |
| 8 | Opponent score | [0, max_score] |

### Reward System

The reward system is designed to encourage effective gameplay:

- **+1.0**: Player scores a point
- **-1.0**: Opponent scores a point  
- **+0.01**: Ball moves closer to player paddle
- **-0.005**: Ball moves away from player paddle
- **+0.05**: Player paddle hits the ball

### Episode Termination

Episodes terminate when:
- Either player reaches the maximum score (default: 11)
- Maximum steps reached (default: 10,000)

## Configuration Options

```python
env = gym.make('Pong-v0', 
               render_mode='human',    # 'human', 'rgb_array', or None
               max_score=11,           # Points needed to win
               max_steps=10000,        # Maximum episode length
               ball_speed=2)           # Initial ball speed
```

## Example Training Scripts

### PPO Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
from pong import register_pong_env

register_pong_env()

# Training environment
train_env = gym.make('Pong-v0')
# Evaluation environment  
eval_env = gym.make('Pong-v0', render_mode='human')

# Evaluation callback
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                           log_path='./logs/', eval_freq=10000,
                           deterministic=True, render=False)

# Train the model
model = PPO('MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=200000, callback=eval_callback)
```

### DQN Training

```python
from stable_baselines3 import DQN
import gymnasium as gym
from pong import register_pong_env

register_pong_env()

env = gym.make('Pong-v0')
model = DQN('MlpPolicy', env, verbose=1, buffer_size=50000,
            learning_starts=10000, target_update_interval=1000)
model.learn(total_timesteps=200000)
model.save("pong_dqn")
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir ./pong_tensorboard/
```

## Performance Tips

1. **Vectorized Environments**: Use multiple parallel environments for faster training
2. **Hyperparameter Tuning**: Experiment with learning rates, network architectures
3. **Reward Shaping**: Modify the reward function based on your training objectives
4. **Curriculum Learning**: Start with slower ball speeds and gradually increase difficulty

## Extending the Environment

You can easily modify the environment for different research purposes:

- **Multi-agent**: Make both paddles trainable
- **Different observations**: Add visual observations (pixel-based)
- **Modified physics**: Change ball dynamics, paddle sizes, etc.
- **Procedural difficulty**: Automatically adjust opponent difficulty

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've activated the virtual environment
2. **Display issues**: For headless servers, use `render_mode=None` or `rgb_array`
3. **Performance**: Use vectorized environments for training, single env for visualization

### Getting Help

- Check the [Gymnasium documentation](https://gymnasium.farama.org/)
- Review [Stable-Baselines3 examples](https://stable-baselines3.readthedocs.io/)
- See our example scripts in the repository

## License

MIT License - see LICENSE file for details.

## References

Based on examples from:
- [gym-kuiper-escape](https://github.com/jdegregorio/gym-kuiper-escape): Custom Gym environment example
- [pygame2gym](https://github.com/sen-pai/pygame2gym): Converting PyGame games to Gym environments
- [Gymnasium Documentation](https://gymnasium.farama.org/): Official API documentation
