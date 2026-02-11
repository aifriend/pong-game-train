"""
Pong Environment Package

Contains gymnasium-compatible Pong environments:
- PongHeadlessEnv: Fast headless environment for training (no pygame)
- PongEnv: Pygame-based environment for visualization
"""

from .pong_headless import PongHeadlessEnv, OpponentType, GameConfig, register_headless_env

__all__ = [
    "PongHeadlessEnv",
    "OpponentType",
    "GameConfig",
    "register_headless_env",
]

