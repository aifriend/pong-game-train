"""Player-controlled paddle."""

from typing import Tuple

from .block import Block
from ..constants import PLAYER_ACCELERATION, PLAYER_DECELERATION, PLAYER_MAX_SPEED


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
        acceleration: float = PLAYER_ACCELERATION,
        deceleration: float = PLAYER_DECELERATION,
        max_speed: float = PLAYER_MAX_SPEED,
    ):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.offset = offset
        self.speed = speed
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.max_speed = max_speed
        self.velocity = 0.0  # Current velocity (gradual)
        self.input_direction = 0  # -1 = up, 0 = none, 1 = down
        self.movement = 0  # Keep for backward compatibility

    def screen_constrain(self) -> None:
        """Keep the paddle within screen bounds."""
        if self.rect.top <= self.offset:
            self.rect.top = self.offset
            # Stop velocity if trying to move up
            if self.velocity < 0:
                self.velocity = 0
        if self.rect.bottom >= self.screen_height - self.offset:
            self.rect.bottom = self.screen_height - self.offset
            # Stop velocity if trying to move down
            if self.velocity > 0:
                self.velocity = 0

    def _is_at_boundary(self) -> Tuple[bool, bool]:
        """Check if paddle is at top or bottom boundary.
        
        Returns:
            Tuple of (at_top, at_bottom)
        """
        at_top = self.rect.top <= self.offset
        at_bottom = self.rect.bottom >= self.screen_height - self.offset
        return at_top, at_bottom

    def update(self, ball_group) -> None:
        """Update player position with acceleration."""
        at_top, at_bottom = self._is_at_boundary()
        
        if self.input_direction != 0:
            # Prevent acceleration if trying to move into a boundary
            if (self.input_direction == -1 and at_top) or (self.input_direction == 1 and at_bottom):
                # At boundary and trying to move into it - stop velocity
                self.velocity = 0
            else:
                # Accelerate in the input direction
                self.velocity += self.input_direction * self.acceleration
                # Clamp to max speed
                self.velocity = max(-self.max_speed, min(self.max_speed, self.velocity))
        else:
            # Decelerate when no input
            if self.velocity > 0:
                self.velocity = max(0, self.velocity - self.deceleration)
            elif self.velocity < 0:
                self.velocity = min(0, self.velocity + self.deceleration)

        self.rect.y += self.velocity
        self.screen_constrain()
