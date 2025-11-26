import random
from typing import Tuple

from .block import Block
from ..constants import OPPONENT_MIN_SPEED, OPPONENT_MAX_SPEED, OPPONENT_RANDOM_RANGE


class Opponent(Block):
    """AI-controlled opponent paddle (or agent-controlled in self-play mode)."""

    def __init__(
        self,
        path: str,
        screen_size: Tuple[int, int],
        offset: int,
        x_pos: int,
        y_pos: int,
        speed: int,
        manual_control: bool = False,
    ):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.offset = offset
        # Use speed parameter or random value within configured range
        self.speed = random.randint(max(speed, OPPONENT_MIN_SPEED), OPPONENT_MAX_SPEED)
        self.movement = 0
        self.manual_control = manual_control

    def screen_constrain(self) -> None:
        """Keep the paddle within screen bounds."""
        if self.rect.top <= self.offset:
            self.rect.top = self.offset
        if self.rect.bottom >= self.screen_height - self.offset:
            self.rect.bottom = self.screen_height - self.offset

    def update(self, ball_group) -> None:
        """Update opponent position based on ball position or manual control."""
        if self.manual_control:
            # Agent-controlled mode: use movement attribute like Player
            self.rect.y += self.movement
        else:
            # AI mode: add randomness to opponent movement for more natural behavior
            rect_flows = self.rect.y + random.randint(-OPPONENT_RANDOM_RANGE, OPPONENT_RANDOM_RANGE)
            if rect_flows < ball_group.sprite.rect.y:
                self.rect.y += self.speed
            if rect_flows > ball_group.sprite.rect.y:
                self.rect.y -= self.speed
        self.screen_constrain()
