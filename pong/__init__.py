"""
Pong Game Package

A classic Pong game implementation with Gymnasium environment wrapper for RL training.

Two environment versions available:
- PongHeadlessEnv: Pure Python environment (recommended for training, no pygame)
- PongEnv: Pygame-based environment (for human play/visualization)

Usage:
    # For training (no pygame needed):
    from pong.env.pong_headless import PongHeadlessEnv
    
    # For visualization (requires pygame):
    from pong.env.pong_gym_env import PongEnv
"""

__version__ = "1.0.0"
__author__ = "Pong Game Developer"

# Lazy imports to avoid pygame dependency unless explicitly needed
# Users should import directly from submodules for best performance:
#   from pong.env.pong_headless import PongHeadlessEnv

def __getattr__(name):
    """Lazy import of environment classes to avoid pygame import on package load."""
    if name == "PongHeadlessEnv":
        from .env.pong_headless import PongHeadlessEnv
        return PongHeadlessEnv
    elif name == "register_headless_env":
        from .env.pong_headless import register_headless_env
        return register_headless_env
    elif name == "PongEnv":
        from .env.pong_gym_env import PongEnv
        return PongEnv
    elif name == "register_pong_env":
        from .env.pong_gym_env import register_pong_env
        return register_pong_env
    raise AttributeError(f"module 'pong' has no attribute '{name}'")

__all__ = [
    "PongHeadlessEnv",
    "register_headless_env",
    "PongEnv",
    "register_pong_env",
]
