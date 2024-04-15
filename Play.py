import sys

import pygame as pygame

from Ball import Ball
from GameObject import GameObject

if __name__ == '__main__':
    # General setup
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    clock = pygame.time.Clock()

    # Main Window (0.75)
    offset = 20
    screen_width = 960
    screen_height = 720
    screen_size = (screen_width, screen_height)
    pygame.display.set_caption('Pong')

    # Game objects
    game_manager, player, opponent = GameObject.get_game_object(
        screen_size=screen_size,
        offset=offset,
        screen_width=screen_width,
        screen_height=screen_height,
    )

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
                if event.key == pygame.K_s and \
                        pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    game_manager, player, opponent = \
                        GameObject.get_game_object(
                            screen_size=screen_size,
                            offset=offset,
                            screen_width=screen_width,
                            screen_height=screen_height,
                            ball_speed=Ball.HIGH_SPEED,
                        )
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
        pygame.display.update()
        clock.tick(120)
