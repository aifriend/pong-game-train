"""Ball class for the Pong game."""

import random
from typing import Tuple

import pygame

from .block import Block
from ..constants import (
    BALL_LOW_SPEED,
    BALL_HIGH_SPEED,
    BALL_SMALL_PAD,
    BALL_LARGE_PAD,
    BALL_ANGLE_20,
    BALL_ANGLE_40,
    PONG_SOUND,
    SCORE_SOUND,
    COLLISION_THRESHOLD,
)


class Ball(Block):
    """Ball object that bounces around the screen."""

    SMALL_PAD = BALL_SMALL_PAD
    LARGE_PAD = BALL_LARGE_PAD
    LOW_SPEED = BALL_LOW_SPEED
    HIGH_SPEED = BALL_HIGH_SPEED
    ANGLE_20 = BALL_ANGLE_20
    ANGLE_40 = BALL_ANGLE_40

    def __init__(
        self,
        path: str,
        screen_size: Tuple[int, int],
        offset: int,
        x_pos: int,
        y_pos: int,
        paddles: pygame.sprite.Group,
        speed_x: int,
        speed_y: int,
        size: int,
    ):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.speed_x = speed_x * random.choice((-1, 1))
        self.speed_y = speed_y * random.choice((-1, 1))
        self.offset = offset
        self.paddles = paddles
        self.score_time = 0

        # Load sounds once (only if mixer is initialized)
        self.pong_sound = None
        self.score_sound = None

        try:
            if pygame.mixer.get_init() is not None:
                if PONG_SOUND.exists():
                    try:
                        self.pong_sound = pygame.mixer.Sound(str(PONG_SOUND))
                    except pygame.error:
                        # Sound file exists but couldn't be loaded
                        pass

                if SCORE_SOUND.exists():
                    try:
                        self.score_sound = pygame.mixer.Sound(str(SCORE_SOUND))
                    except pygame.error:
                        # Sound file exists but couldn't be loaded
                        pass
        except pygame.error:
            # Mixer not initialized, sounds will be disabled
            pass

    def update(self, game_manager) -> None:
        """Update ball position and check collisions."""
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        self.collisions()

    def collisions(self) -> None:
        """Handle collisions with walls and paddles."""
        # Wall collisions (top and bottom)
        if self.rect.top <= self.offset or self.rect.bottom >= self.screen_height - self.offset:
            if self.pong_sound:
                self.pong_sound.play()
            self.speed_y *= -1

        # Paddle collisions
        collided_paddles = pygame.sprite.spritecollide(self, self.paddles, False)
        if collided_paddles:
            if self.pong_sound:
                self.pong_sound.play()
            collision_paddle = collided_paddles[0].rect

            # Horizontal collisions (left/right sides of paddles)
            if (
                abs(self.rect.right - collision_paddle.left) < COLLISION_THRESHOLD
                and self.speed_x > 0
            ):
                self.speed_x *= -1
            if (
                abs(self.rect.left - collision_paddle.right) < COLLISION_THRESHOLD
                and self.speed_x < 0
            ):
                self.speed_x *= -1

            # Vertical collisions (top/bottom of paddles)
            if (
                abs(self.rect.top - collision_paddle.bottom) < COLLISION_THRESHOLD
                and self.speed_y < 0
            ):
                self.rect.top = collision_paddle.bottom
                self.speed_y *= -1
            if (
                abs(self.rect.bottom - collision_paddle.top) < COLLISION_THRESHOLD
                and self.speed_y > 0
            ):
                self.rect.bottom = collision_paddle.top
                self.speed_y *= -1

    def reset_ball(self) -> None:
        """Reset ball to center and randomize direction."""
        self.speed_x *= random.choice((-1, 1))
        self.speed_y *= random.choice((-1, 1))
        self.score_time = pygame.time.get_ticks()
        self.rect.center = (self.screen_width / 2, self.screen_height / 2)
        if self.score_sound:
            self.score_sound.play()
