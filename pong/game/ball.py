"""Ball class for the graphical Pong game."""

import math
import random
from typing import Tuple

import pygame

from .block import Block
from .physics import bounce_velocity
from ..constants import (
    BALL_LOW_SPEED,
    BALL_HIGH_SPEED,
    BALL_SMALL_PAD,
    BALL_LARGE_PAD,
    BALL_ANGLE_20,
    BALL_ANGLE_40,
    BALL_ACCELERATION,
    BALL_MAX_SPEED_MULTIPLIER,
    BALL_MAX_BOUNCE_ANGLE,
    PADDLE_SPIN_ANGLE,
    PLAYER_MAX_SPEED,
    SERVE_DELAY_MS,
    PONG_SOUND,
    SCORE_SOUND,
)


class Ball(Block):
    """Ball with smooth movement, serve pauses, and controlled rebounds."""

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
        self.base_speed = abs(speed_x)
        self.offset = offset
        self.paddles = paddles
        self._x = float(self.rect.centerx)
        self._y = float(self.rect.centery)
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.score_time = pygame.time.get_ticks()
        self._set_serve_velocity()

        self.pong_sound = None
        self.score_sound = None
        try:
            if pygame.mixer.get_init() is not None:
                if PONG_SOUND.exists():
                    try:
                        self.pong_sound = pygame.mixer.Sound(str(PONG_SOUND))
                    except pygame.error:
                        pass
                if SCORE_SOUND.exists():
                    try:
                        self.score_sound = pygame.mixer.Sound(str(SCORE_SOUND))
                    except pygame.error:
                        pass
        except pygame.error:
            pass

    def _set_serve_velocity(self) -> None:
        angle = math.radians(random.uniform(-35, 35))
        direction = random.choice((-1, 1))
        self.speed_x = direction * self.base_speed * math.cos(angle)
        self.speed_y = self.base_speed * math.sin(angle)

    def _sync_position(self) -> None:
        self.rect.center = (round(self._x), round(self._y))

    def update(self, game_manager) -> None:
        """Update ball position and check collisions after the serve pause."""
        if pygame.time.get_ticks() - self.score_time < SERVE_DELAY_MS:
            return
        previous = self.rect.copy()
        self._x += self.speed_x
        self._y += self.speed_y
        self._sync_position()
        self.collisions(previous)

    def collisions(self, previous: pygame.Rect) -> None:
        """Handle wall and front-face paddle collisions."""
        if self.rect.top <= self.offset and self.speed_y < 0:
            self.rect.top = self.offset
            self.speed_y = abs(self.speed_y)
            if self.pong_sound:
                self.pong_sound.play()
        elif self.rect.bottom >= self.screen_height - self.offset and self.speed_y > 0:
            self.rect.bottom = self.screen_height - self.offset
            self.speed_y = -abs(self.speed_y)
            if self.pong_sound:
                self.pong_sound.play()

        for paddle in self.paddles:
            if not self.rect.colliderect(paddle.rect):
                continue
            moving_right = self.speed_x > 0
            crossed_front = (
                moving_right
                and previous.right <= paddle.rect.left
                and self.rect.right >= paddle.rect.left
            ) or (
                not moving_right
                and previous.left >= paddle.rect.right
                and self.rect.left <= paddle.rect.right
            )
            if not crossed_front:
                continue

            if moving_right:
                distance = self.rect.right - previous.right
                travel = paddle.rect.left - previous.right
                self.rect.right = paddle.rect.left
                direction = -1
            else:
                distance = previous.left - self.rect.left
                travel = previous.left - paddle.rect.right
                self.rect.left = paddle.rect.right
                direction = 1

            fraction = max(0.0, min(1.0, travel / distance)) if distance else 1.0
            contact_y = previous.centery + fraction * (
                self.rect.centery - previous.centery
            )
            hit_offset = (contact_y - paddle.rect.centery) / (
                paddle.rect.height / 2 + self.rect.height / 2
            )
            self.speed_x, self.speed_y = bounce_velocity(
                self.speed_x,
                self.speed_y,
                hit_offset,
                direction,
                getattr(paddle, "velocity", 0.0),
                PLAYER_MAX_SPEED,
                BALL_ACCELERATION,
                self.base_speed * BALL_MAX_SPEED_MULTIPLIER,
                BALL_MAX_BOUNCE_ANGLE,
                PADDLE_SPIN_ANGLE,
            )
            if self.pong_sound:
                self.pong_sound.play()
            break

        self._x = float(self.rect.centerx)
        self._y = float(self.rect.centery)

    def reset_ball(self) -> None:
        """Reset to center with a fresh angle and a short serve pause."""
        self._x = self.screen_width / 2
        self._y = self.screen_height / 2
        self._sync_position()
        self._set_serve_velocity()
        self.score_time = pygame.time.get_ticks()
        if self.score_sound:
            self.score_sound.play()
