"""Factory for creating game objects."""

import pygame
from typing import Tuple

from .ball import Ball
from .game_manager import GameManager
from .opponent import Opponent
from .player import Player
from ..constants import PADDLE_IMAGE, BALL_IMAGE


class GameObject:
    """Factory class for creating game objects."""

    @staticmethod
    def get_game_object(
        screen_size: Tuple[int, int],
        offset: int,
        screen_width: int,
        screen_height: int,
        ball_speed: int = Ball.LOW_SPEED,
        pad_size: int = Ball.SMALL_PAD,
        opponent_manual_control: bool = False,
        headless: bool = False,
    ) -> Tuple[GameManager, Player, Opponent]:
        """
        Create and return all game objects.

        Args:
            screen_size: Tuple of (width, height)
            offset: Screen border offset
            screen_width: Screen width
            screen_height: Screen height
            ball_speed: Initial ball speed
            pad_size: Ball size modifier (currently unused, reserved for future use)
            opponent_manual_control: If True, opponent uses movement attribute instead of AI
            headless: If True, use Surface instead of display (for training without GUI)

        Returns:
            Tuple of (GameManager, Player, Opponent)
        """
        center_x = screen_width / 2
        center_y = screen_height / 2

        player = Player(
            str(PADDLE_IMAGE), screen_size, offset, screen_width - offset, center_y, ball_speed
        )
        opponent = Opponent(
            str(PADDLE_IMAGE),
            screen_size,
            offset,
            offset,
            center_y,
            ball_speed,
            manual_control=opponent_manual_control,
        )
        paddle_group = pygame.sprite.Group()
        paddle_group.add(player)
        paddle_group.add(opponent)

        ball = Ball(
            str(BALL_IMAGE),
            screen_size,
            offset,
            center_x,
            center_y,
            paddle_group,
            speed_x=ball_speed,
            speed_y=ball_speed,
            size=pad_size,
        )
        ball_sprite = pygame.sprite.GroupSingle()
        ball_sprite.add(ball)

        game_manager = GameManager(
            screen_size, offset, ball_sprite, paddle_group, headless=headless
        )

        return game_manager, player, opponent
