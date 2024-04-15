import pygame as pygame

from Ball import Ball
from GameManager import GameManager
from Opponent import Opponent
from Player import Player


class GameObject:

    @staticmethod
    def get_game_object(
            screen_size,
            offset,
            screen_width,
            screen_height,
            ball_speed=Ball.LOW_SPEED,
            pad_size=Ball.SMALL_PAD
    ):
        mdl = lambda x: x / 2

        _player = Player(
            'resources/paddle.png',
            screen_size,
            offset,
            screen_width - offset,
            mdl(screen_height),
            ball_speed
        )
        _opponent = Opponent(
            'resources/paddle.png',
            screen_size,
            offset,
            offset,
            mdl(screen_height),
            ball_speed
        )
        paddle_group = pygame.sprite.Group()
        paddle_group.add(_player)
        paddle_group.add(_opponent)
        ball = Ball(
            'resources/ball.png',
            screen_size,
            offset,
            mdl(screen_width),
            mdl(screen_height),
            paddle_group,
            speed_x=ball_speed,
            speed_y=ball_speed,
            size=pad_size
        )
        ball_sprite = pygame.sprite.GroupSingle()
        ball_sprite.add(ball)

        _game_manager = GameManager(
            screen_size, offset, ball_sprite, paddle_group)

        return _game_manager, _player, _opponent
