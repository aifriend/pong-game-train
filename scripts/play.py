"""Main game entry point."""

import sys

import pygame

from pong.game.ball import Ball
from pong.game.game_object import GameObject
from pong.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_SIZE,
    OFFSET,
    FPS,
    AUDIO_FREQUENCY,
    AUDIO_SIZE,
    AUDIO_CHANNELS,
    AUDIO_BUFFER,
)


def main() -> None:
    """Main game loop."""
    # General setup
    pygame.mixer.pre_init(AUDIO_FREQUENCY, AUDIO_SIZE, AUDIO_CHANNELS, AUDIO_BUFFER)
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption("Pong")

    # Game objects
    game_manager, player, opponent = GameObject.get_game_object(
        screen_size=SCREEN_SIZE,
        offset=OFFSET,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    player.movement -= player.speed
                elif event.key == pygame.K_DOWN:
                    player.movement += player.speed
                elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Reset game with high speed
                    game_manager, player, opponent = GameObject.get_game_object(
                        screen_size=SCREEN_SIZE,
                        offset=OFFSET,
                        screen_width=SCREEN_WIDTH,
                        screen_height=SCREEN_HEIGHT,
                        ball_speed=Ball.HIGH_SPEED,
                    )
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    player.movement += player.speed
                elif event.key == pygame.K_DOWN:
                    player.movement -= player.speed

        # Background
        game_manager.background()

        # Run the game
        game_manager.run_game()

        # Rendering
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
