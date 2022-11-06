import sys

import pygame as pygame

from Ball import Ball
from GameManager import GameManager
from Opponent import Opponent
from Player import Player

if __name__ == '__main__':
    # General setup
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    clock = pygame.time.Clock()

    # Main Window (0.75)
    game_speed = 2
    offset = 20
    screen_width = 960
    screen_height = 720
    screen_size = (screen_width, screen_height)
    mdl = lambda x: x/2
    pygame.display.set_caption('Pong')

    # Game objects
    player = Player(
        'resources/paddle.png',
        screen_size,
        offset,
        screen_width - offset,
        mdl(screen_height),
        game_speed)
    opponent = Opponent(
        'resources/paddle.png',
        screen_size,
        offset,
        offset,
        mdl(screen_width),
        game_speed)
    paddle_group = pygame.sprite.Group()
    paddle_group.add(player)
    paddle_group.add(opponent)
    ball = Ball(
        'resources/ball.png',
        screen_size,
        offset,
        mdl(screen_width),
        mdl(screen_height),
        game_speed,
        game_speed,
        paddle_group)
    ball_sprite = pygame.sprite.GroupSingle()
    ball_sprite.add(ball)

    game_manager = GameManager(screen_size, offset, ball_sprite, paddle_group)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.movement -= player.speed
                if event.key == pygame.K_DOWN:
                    player.movement += player.speed
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    player.movement += player.speed
                if event.key == pygame.K_DOWN:
                    player.movement -= player.speed

        # Background Stuff
        game_manager.background()

        # Run the game
        game_manager.run_game()

        # Rendering
        pygame.display.flip()
        clock.tick(120)
