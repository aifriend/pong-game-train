"""Base class for game objects."""

import pygame
from pathlib import Path


class Block(pygame.sprite.Sprite):
    """Base class for all game objects (ball, paddles)."""

    def __init__(self, path: str, x_pos: int, y_pos: int):
        super().__init__()
        if not Path(path).exists():
            raise FileNotFoundError(f"Resource not found: {path}")
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect(center=(x_pos, y_pos))
