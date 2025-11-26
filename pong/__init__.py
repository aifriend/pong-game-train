"""
Pong Game Package

A classic Pong game implementation with Gymnasium environment wrapper for RL training.

Two environment versions available:
- PongEnv: Full pygame-based environment (requires display)
- PongHeadlessEnv: Pure Python environment (no pygame needed for simulation)
"""

__version__ = "1.0.0"
__author__ = "Pong Game Developer"

# Import headless environment first (no pygame dependency)
from .env.pong_headless import PongHeadlessEnv, register_headless_env

# Try to import pygame-based environment (may fail on headless systems)
try:
    from .env.pong_gym_env import PongEnv, register_pong_env

    _PYGAME_AVAILABLE = True
except ImportError:
    PongEnv = None
    register_pong_env = None
    _PYGAME_AVAILABLE = False

__all__ = [
    "PongHeadlessEnv",
    "register_headless_env",
    "PongEnv",
    "register_pong_env",
]

# Note: For headless training, use PongHeadlessEnv
# For rendering, use PongEnv with render_mode='human'
