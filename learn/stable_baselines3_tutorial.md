# Stable-Baselines3 Complete Tutorial

A comprehensive guide to using Stable-Baselines3 with the Pong environment.

## Table of Contents
1. [What is Stable-Baselines3?](#what-is-stable-baselines3)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Basic Usage - Step by Step](#basic-usage)
5. [Algorithms: PPO vs DQN](#algorithms)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)

---

## What is Stable-Baselines3?

Stable-Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It provides:

- **Production-ready implementations** of state-of-the-art RL algorithms
- **Multiple algorithms**: PPO, DQN, A2C, SAC, TD3, and more
- **Built-in utilities**: Callbacks, monitoring, evaluation
- **TensorBoard integration** for visualization
- **Easy to use API** that follows Gymnasium standards

---

## Installation

```bash
pip install stable-baselines3[extra]
# The [extra] includes additional dependencies like tensorboard
```

Or with your requirements:
```bash
pip install stable-baselines3>=2.0.0 tensorboard>=2.10.0
```

---

## Core Concepts

### 1. Policy Types

- **`MlpPolicy`**: Multi-layer perceptron (for vector observations like your Pong)
- **`CnnPolicy`**: Convolutional neural network (for image observations)
- **`MultiInputPolicy`**: For mixed observation types

### 2. Model Creation

```python
from stable_baselines3 import PPO

model = PPO(
    policy='MlpPolicy',  # What type of network
    env=env,              # The environment
    verbose=1             # Print training info
)
```

### 3. Training

```python
model.learn(total_timesteps=100000)
```

### 4. Using Trained Models

```python
action, _ = model.predict(observation, deterministic=True)
```

---

## Basic Usage - Step by Step

### Step 1: Create/Register Environment

```python
import gymnasium as gym
from pong import register_pong_env

# Register your custom environment
register_pong_env()

# Create environment
env = gym.make('Pong-v0', max_score=11, max_steps=5000)
```

### Step 2: Create a Model

**PPO Example:**
```python
from stable_baselines3 import PPO

model = PPO(
    'MlpPolicy',           # Policy type
    env,                   # Environment
    learning_rate=3e-4,    # How fast to learn
    n_steps=2048,          # Steps per update
    batch_size=64,         # Batch size for training
    n_epochs=10,           # How many times to train on same data
    gamma=0.99,            # Discount factor
    verbose=1,             # Print info
    tensorboard_log="./logs/"  # Log directory
)
```

**DQN Example:**
```python
from stable_baselines3 import DQN

model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=50000,      # Replay buffer size
    learning_starts=1000,   # Steps before learning starts
    batch_size=32,
    target_update_interval=1000,  # Update target network every N steps
    exploration_final_eps=0.05,   # Final exploration rate
    verbose=1,
    tensorboard_log="./logs/"
)
```

### Step 3: Train the Model

```python
# Simple training
model.learn(total_timesteps=100000)

# With callbacks (recommended)
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Wrap env for monitoring
eval_env = Monitor(gym.make('Pong-v0'))

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=5000,        # Evaluate every 5000 steps
    deterministic=True,
    render=False
)

# Train with callback
model.learn(total_timesteps=100000, callback=eval_callback)
```

### Step 4: Save the Model

```python
model.save("pong_agent")
# Saves as: pong_agent.zip
```

### Step 5: Load and Use the Model

```python
# Load
model = PPO.load("pong_agent")

# Use
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

---

## Algorithms: PPO vs DQN

### PPO (Proximal Policy Optimization)

**When to use:**
- ✅ Continuous or discrete actions
- ✅ On-policy (uses current policy's data)
- ✅ Generally more stable
- ✅ Good for your Pong environment

**Key parameters:**
```python
PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,      # Learning rate
    n_steps=2048,            # Steps collected per update
    batch_size=64,           # Minibatch size
    n_epochs=10,             # Training epochs per update
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_range=0.2,          # PPO clip range
    ent_coef=0.01,           # Entropy coefficient (exploration)
    vf_coef=0.5,             # Value function coefficient
)
```

### DQN (Deep Q-Network)

**When to use:**
- ✅ Discrete actions only
- ✅ Off-policy (can use old data)
- ✅ Good for value-based learning
- ✅ Works well with experience replay

**Key parameters:**
```python
DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-4,
    buffer_size=50000,        # Replay buffer size
    learning_starts=1000,     # Steps before learning
    batch_size=32,
    tau=1.0,                  # Soft update coefficient
    gamma=0.99,
    train_freq=4,             # Train every N steps
    gradient_steps=1,         # Gradient steps per update
    target_update_interval=1000,  # Hard update frequency
    exploration_fraction=0.1,      # Exploration schedule
    exploration_initial_eps=1.0,   # Initial epsilon
    exploration_final_eps=0.05,    # Final epsilon
)
```

---

## Advanced Features

### 1. Vectorized Environments (Parallel Training)

Train on multiple environments simultaneously:

```python
from stable_baselines3.common.env_util import make_vec_env

# Create 4 parallel environments
vec_env = make_vec_env(
    lambda: gym.make('Pong-v0', max_score=5),
    n_envs=4  # Number of parallel environments
)

model = PPO('MlpPolicy', vec_env, verbose=1)
model.learn(total_timesteps=100000)
# Trains 4x faster!
```

### 2. Callbacks

**EvalCallback** - Evaluate during training:
```python
from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True,
    render=False,
    n_eval_episodes=10  # Number of episodes to evaluate
)
```

**StopTrainingOnRewardThreshold** - Early stopping:
```python
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    StopTrainingOnRewardThreshold
)

# Stop when reward threshold reached
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=4.0,
    verbose=1
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    ...
)
```

**Custom Callback**:
```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        # Called at each step
        if self.locals.get('dones')[0]:  # Episode ended
            reward = self.locals.get('rewards')[0]
            self.episode_rewards.append(reward)
            print(f"Episode reward: {reward}")
        return True  # Continue training

model.learn(total_timesteps=100000, callback=CustomCallback())
```

### 3. Monitoring

```python
from stable_baselines3.common.monitor import Monitor

# Wrap environment to log statistics
monitored_env = Monitor(env, './logs/')

# Training logs saved automatically
model = PPO('MlpPolicy', monitored_env)
model.learn(total_timesteps=100000)
```

### 4. TensorBoard Integration

```python
model = PPO(
    'MlpPolicy',
    env,
    tensorboard_log="./tensorboard_logs/"
)

model.learn(total_timesteps=100000)

# View logs:
# tensorboard --logdir ./tensorboard_logs/
```

### 5. Custom Policies

```python
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

# Custom network architecture
class CustomNetwork(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
    
    def forward(self, obs):
        return self.net(obs)

# Use custom network
model = PPO(
    ActorCriticPolicy,
    env,
    policy_kwargs=dict(
        features_extractor_class=CustomNetwork,
        features_extractor_kwargs=dict(features_dim=9)  # Your obs dim
    )
)
```

---

## Best Practices

### 1. Hyperparameter Tuning

Start with defaults, then tune:

```python
# Conservative learning
model = PPO('MlpPolicy', env, learning_rate=1e-4, n_steps=1024)

# Aggressive learning
model = PPO('MlpPolicy', env, learning_rate=5e-4, n_steps=4096)
```

### 2. Environment Configuration

```python
# Shorter games for faster training
train_env = gym.make('Pong-v0', max_score=5, max_steps=5000)

# Longer games for evaluation
eval_env = gym.make('Pong-v0', max_score=11, max_steps=10000)
```

### 3. Training Schedule

```python
# Train in stages
model.learn(total_timesteps=50000)   # Initial training
model.save("pong_agent_stage1")

# Continue training
model.learn(total_timesteps=50000)   # Fine-tuning
model.save("pong_agent_final")
```

### 4. Evaluation

```python
import numpy as np

def evaluate_model(model, env, n_episodes=10):
    episode_rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
    
    return {
        'mean': np.mean(episode_rewards),
        'std': np.std(episode_rewards),
        'min': np.min(episode_rewards),
        'max': np.max(episode_rewards)
    }
```

---

## Common Patterns

### Pattern 1: Complete Training Pipeline

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from pong import register_pong_env

# 1. Register environment
register_pong_env()

# 2. Create vectorized training env
train_env = make_vec_env(
    lambda: gym.make('Pong-v0', max_score=5),
    n_envs=4
)

# 3. Create evaluation env
eval_env = Monitor(gym.make('Pong-v0', max_score=5))

# 4. Create callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=5000,
    deterministic=True
)

# 5. Create and train model
model = PPO(
    'MlpPolicy',
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

model.learn(total_timesteps=200000, callback=eval_callback)

# 6. Save final model
model.save("./models/pong_ppo_final")
```

### Pattern 2: Resume Training

```python
# Load existing model
model = PPO.load("./models/pong_ppo_final", env=train_env)

# Continue training
model.learn(total_timesteps=100000, reset_num_timesteps=False)
model.save("./models/pong_ppo_final_v2")
```

### Pattern 3: Compare Algorithms

```python
algorithms = {
    'PPO': PPO,
    'DQN': DQN
}

results = {}
for name, algo_class in algorithms.items():
    model = algo_class('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save(f"./models/pong_{name.lower()}")
    
    # Evaluate
    results[name] = evaluate_model(model, eval_env)
```

---

## Quick Reference

### Essential Imports
```python
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
```

### Model Lifecycle
```python
# Create
model = PPO('MlpPolicy', env)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("model_name")

# Load
model = PPO.load("model_name")

# Use
action, _ = model.predict(obs, deterministic=True)
```

### Key Hyperparameters

**PPO:**
- `learning_rate`: 1e-4 to 5e-4
- `n_steps`: 1024 to 4096
- `batch_size`: 32 to 256
- `gamma`: 0.9 to 0.99

**DQN:**
- `learning_rate`: 1e-4 to 1e-3
- `buffer_size`: 10000 to 1000000
- `batch_size`: 32 to 128
- `exploration_final_eps`: 0.01 to 0.1

---

## Next Steps

1. Run your existing script: `python scripts/train_pong_agent.py`
2. Monitor with TensorBoard: `tensorboard --logdir ./pong_tensorboard/`
3. Experiment with hyperparameters
4. Try different algorithms (A2C, SAC for continuous actions)
5. Implement custom callbacks for your needs

