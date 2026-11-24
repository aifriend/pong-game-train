"""Main graphical game entry point."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pygame

from pong.game.ball import Ball
from pong.game.game_object import GameObject
from pong.game.opponent import Opponent
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


def create_game(difficulty: str, ball_speed: int = Ball.LOW_SPEED):
    return GameObject.get_game_object(
        screen_size=SCREEN_SIZE,
        offset=OFFSET,
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
        ball_speed=ball_speed,
        opponent_difficulty=difficulty,
    )


def main() -> None:
    """Run Pong with human controls and a predictive opponent."""
    parser = argparse.ArgumentParser(description="Play realistic Pong")
    parser.add_argument(
        "--difficulty",
        choices=list(Opponent.DIFFICULTIES),
        default="expert",
        help="Opponent level (default: expert)",
    )
    args = parser.parse_args()

    pygame.mixer.pre_init(AUDIO_FREQUENCY, AUDIO_SIZE, AUDIO_CHANNELS, AUDIO_BUFFER)
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption(f"Pong — {args.difficulty.title()} AI")
    game_manager, player, opponent = create_game(args.difficulty)

    print("Controls: UP/DOWN or W/S to move, Shift+S for a fast reset, ESC to quit.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif (
                    event.key == pygame.K_s
                    and pygame.key.get_mods() & pygame.KMOD_SHIFT
                ):
                    game_manager, player, opponent = create_game(
                        args.difficulty, Ball.HIGH_SPEED
                    )
                elif event.key in (pygame.K_UP, pygame.K_w):
                    player.input_direction = -1
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    player.input_direction = 1
            elif event.type == pygame.KEYUP:
                if (
                    event.key in (pygame.K_UP, pygame.K_w)
                    and player.input_direction == -1
                ):
                    player.input_direction = 0
                elif (
                    event.key in (pygame.K_DOWN, pygame.K_s)
                    and player.input_direction == 1
                ):
                    player.input_direction = 0

        game_manager.background()
        game_manager.run_game()
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
