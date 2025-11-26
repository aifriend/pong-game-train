"""
Centralized configuration for DQN training on Pong.
All hyperparameters and settings in one place for easy experimentation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    env_name: str = "Pong-v0"
    render_mode: Optional[str] = None  # 'human' for visualization, None for speed
    max_score: int = 11
    max_steps: int = 10000
    self_play: bool = True  # Agent vs agent training


@dataclass
class AgentConfig:
    """DQN Agent hyperparameters."""

    # Actions
    possible_actions: List[int] = field(default_factory=lambda: [0, 1, 2])
    observation_dim: int = 9

    # Memory
    starting_mem_len: int = 5000  # Minimum experiences before learning
    max_mem_len: int = 100000  # Maximum replay buffer size

    # Learning (optimized v2 - improved learning signal)
    learning_rate: float = 0.001  # Doubled for faster learning
    gamma: float = 0.99  # Discount factor
    batch_size: int = 128  # Doubled for stable gradients
    learn_every: int = 8  # Learn every N steps (more diverse experience)

    # Exploration handled by noisy networks (no epsilon needed)

    # Target network
    target_update_freq: int = 5000  # Less frequent updates for stability

    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    max_episodes: int = 100000
    save_interval: int = 50  # Save plot every N episodes
    checkpoint_interval: int = 100  # Save model every N episodes
    print_interval: int = 10  # Print stats every N episodes

    # Directories
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "./tensorboard_dqn/"

    # Self-play (exploration handled by noisy networks, no epsilon needed)


@dataclass
class Config:
    """Complete configuration."""

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Create directories."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.tensorboard_dir, exist_ok=True)


# Default configuration
DEFAULT_CONFIG = Config()


# Pre-defined configurations for different scenarios
FAST_TRAINING_CONFIG = Config(
    env=EnvironmentConfig(render_mode=None, self_play=True),
    agent=AgentConfig(
        starting_mem_len=2000,
        max_mem_len=50000,
        learning_rate=0.001,
    ),
    training=TrainingConfig(
        max_episodes=50000,
        checkpoint_interval=50,
    ),
)

DEBUG_CONFIG = Config(
    env=EnvironmentConfig(render_mode="human", self_play=True),
    agent=AgentConfig(
        starting_mem_len=1000,
        max_mem_len=10000,
    ),
    training=TrainingConfig(
        max_episodes=1000,
        print_interval=1,
        checkpoint_interval=100,
    ),
)

PRODUCTION_CONFIG = Config(
    env=EnvironmentConfig(render_mode=None, self_play=True),
    agent=AgentConfig(
        starting_mem_len=10000,
        max_mem_len=200000,
        learning_rate=0.0003,
        gamma=0.99,
    ),
    training=TrainingConfig(
        max_episodes=200000,
        checkpoint_interval=500,
    ),
)
