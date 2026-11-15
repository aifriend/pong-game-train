"""
Pong Game Package

A classic Pong game implementation with Gymnasium environment wrapper for RL training.

The training environment is PongHeadlessEnv, a pure Python environment with no
pygame dependency. Human play lives in scripts/play.py.

Usage:
    from pong.env.pong_headless import PongHeadlessEnv
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
    raise AttributeError(f"module 'pong' has no attribute '{name}'")


__all__ = [
    "PongHeadlessEnv",
    "register_headless_env",
]
