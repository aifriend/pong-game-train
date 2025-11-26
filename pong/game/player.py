"""Player-controlled paddle."""

from typing import Tuple

from .block import Block


class Player(Block):
    """Player-controlled paddle."""

    def __init__(
        self,
        path: str,
        screen_size: Tuple[int, int],
        offset: int,
        x_pos: int,
        y_pos: int,
        speed: int,
    ):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.offset = offset
        self.speed = speed
        self.movement = 0

    def screen_constrain(self) -> None:
        """Keep the paddle within screen bounds."""
        if self.rect.top <= self.offset:
            self.rect.top = self.offset
        if self.rect.bottom >= self.screen_height - self.offset:
            self.rect.bottom = self.screen_height - self.offset

    def update(self, ball_group) -> None:
        """Update player position based on movement."""
        self.rect.y += self.movement
        self.screen_constrain()
