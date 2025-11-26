"""
DQN trainer package.
Contains DQN agent implementation using PyTorch for training on Pong environment.
"""

from .the_agent_pytorch import Agent
from .agent_memory import Memory

__all__ = ["Agent", "Memory"]
