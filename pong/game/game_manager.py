"""Game manager for handling game state and rendering."""

import pygame
from typing import Tuple

from .point import Point
from ..constants import (
    BG_COLOR,
    ACCENT_COLOR,
    FONT_PATH,
    FALLBACK_FONT,
    FONT_SIZE_BASIC,
    FONT_SIZE_SCORE,
    SCORE_RESOLUTION,
    OPPONENT_SCORE_X_MULTIPLIER,
    PLAYER_SCORE_X_MULTIPLIER,
    SCORE_Y_MULTIPLIER,
)


class GameManager:
    """Manages game state, rendering, and score."""

    def __init__(
        self,
        screen_size: Tuple[int, int],
        offset: int,
        ball_group: pygame.sprite.GroupSingle,
        paddle_group: pygame.sprite.Group,
        headless: bool = False,
    ):
        self.offset = offset
        self.bg_color = pygame.Color(BG_COLOR)
        self.accent_color = ACCENT_COLOR
        self.screen_width, self.screen_height = screen_size
        self.headless = headless

        # Create screen - use Surface for headless mode to avoid display issues
        if headless:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        self.player_score = 0
        self.opponent_score = 0
        self.ball_group = ball_group
        self.paddle_group = paddle_group
        self.playground_strip_up = pygame.Rect(0, 0, self.screen_width, self.offset)
        self.playground_strip_down = pygame.Rect(
            0, self.screen_height - self.offset, self.screen_width, self.offset
        )

        # Load fonts with fallback
        try:
            self.basic_font = pygame.font.Font(str(FONT_PATH), FONT_SIZE_BASIC)
        except (FileNotFoundError, pygame.error):
            self.basic_font = pygame.font.Font(FALLBACK_FONT, FONT_SIZE_BASIC)

        try:
            self.score_font = pygame.font.Font(str(FONT_PATH), FONT_SIZE_SCORE)
        except (FileNotFoundError, pygame.error):
            self.score_font = pygame.font.Font(FALLBACK_FONT, FONT_SIZE_SCORE)

    def run_game(self) -> None:
        """Run one frame of the game."""
        # Drawing the game objects
        self.paddle_group.draw(self.screen)
        self.ball_group.draw(self.screen)

        # Updating the game objects
        self.paddle_group.update(self.ball_group)
        self.ball_group.update(self)
        self.check_score()
        self.draw_score()

    def check_score(self) -> None:
        """Check if ball has scored and reset if needed."""
        if self.ball_group.sprite.rect.right >= self.screen_width:
            self.opponent_score += 1
            self.ball_group.sprite.reset_ball()
        if self.ball_group.sprite.rect.left <= 0:
            self.player_score += 1
            self.ball_group.sprite.reset_ball()

    @staticmethod
    def draw_dashed_line(
        surf: pygame.Surface,
        color: Tuple[int, int, int],
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        width: int = 20,
        dash_length: int = 10,
    ) -> None:
        """Draw a dashed line between two points."""
        origin = Point(start_pos)
        target = Point(end_pos)
        displacement = target - origin
        length = len(displacement)

        if length == 0:
            return

        slope = displacement / length

        for index in range(0, length // dash_length, 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pygame.draw.line(surf, color, start.get(), end.get(), width)

    def draw_score(self) -> None:
        """Draw player and opponent scores on screen."""
        player_score_text = self.score_font.render(str(self.player_score), True, self.accent_color)
        opponent_score_text = self.score_font.render(
            str(self.opponent_score), True, self.accent_color
        )

        width_offset = self.screen_width / SCORE_RESOLUTION
        opponent_score_rect = opponent_score_text.get_rect(
            center=(
                width_offset * OPPONENT_SCORE_X_MULTIPLIER,
                self.screen_height / SCORE_RESOLUTION * SCORE_Y_MULTIPLIER,
            )
        )
        player_score_rect = player_score_text.get_rect(
            center=(
                width_offset * PLAYER_SCORE_X_MULTIPLIER,
                self.screen_height / SCORE_RESOLUTION * SCORE_Y_MULTIPLIER,
            )
        )

        self.screen.blit(player_score_text, player_score_rect)
        self.screen.blit(opponent_score_text, opponent_score_rect)

    def background(self) -> None:
        """Draw the game background."""
        self.screen.fill(self.bg_color)
        pygame.draw.rect(self.screen, self.accent_color, self.playground_strip_up)
        pygame.draw.rect(self.screen, self.accent_color, self.playground_strip_down)
        self.draw_dashed_line(
            self.screen,
            self.accent_color,
            start_pos=(self.screen_width / 2 - 1, self.offset),
            end_pos=(self.screen_width / 2 - 1, self.screen_height),
            width=self.offset - 5,
            dash_length=self.offset,
        )
