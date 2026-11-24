"""Predictive AI-controlled opponent paddle."""

from typing import Tuple

from .block import Block
from .physics import predict_intercept
from ..constants import PLAYER_MAX_SPEED


class Opponent(Block):
    """AI opponent with tracking, prediction, and expert shot placement."""

    DIFFICULTIES = {
        "slow": (0.40, 35.0),
        "beginner": (0.55, 28.0),
        "medium": (0.65, 22.0),
        "normal": (0.70, 20.0),
        "reactive": (0.70, 18.0),
        "expert": (1.0, 1.0),
    }

    def __init__(
        self,
        path: str,
        screen_size: Tuple[int, int],
        offset: int,
        x_pos: int,
        y_pos: int,
        speed: int,
        manual_control: bool = False,
        difficulty: str = "expert",
    ):
        super().__init__(path, x_pos, y_pos)
        if difficulty not in self.DIFFICULTIES:
            raise ValueError(f"Unknown opponent difficulty: {difficulty}")
        self.screen_width, self.screen_height = screen_size
        self.offset = offset
        speed_multiplier, self.dead_zone = self.DIFFICULTIES[difficulty]
        self.difficulty = difficulty
        self.speed = PLAYER_MAX_SPEED * speed_multiplier
        self.movement = 0
        self.velocity = 0.0
        self.manual_control = manual_control

    def screen_constrain(self) -> None:
        """Keep the paddle within screen bounds."""
        self.rect.top = max(self.rect.top, self.offset)
        self.rect.bottom = min(self.rect.bottom, self.screen_height - self.offset)

    def _target_y(self, ball) -> float:
        if ball.speed_x >= 0:
            return self.screen_height / 2

        predict = self.difficulty in ("reactive", "expert")
        if self.difficulty == "reactive":
            predict = ball.rect.centerx < self.screen_width * 0.30
        if not predict:
            return float(ball.rect.centery)

        target_x = self.rect.right + ball.rect.width / 2
        target_y, _ = predict_intercept(
            ball.rect.centerx,
            ball.rect.centery,
            ball.speed_x,
            ball.speed_y,
            target_x,
            self.offset + ball.rect.height / 2,
            self.screen_height - self.offset - ball.rect.height / 2,
        )
        if self.difficulty == "expert":
            other_paddle = next(
                (paddle for paddle in ball.paddles if paddle is not self), None
            )
            other_y = (
                other_paddle.rect.centery
                if other_paddle is not None
                else self.screen_height / 2
            )
            contact_offset = 0.85 if other_y < self.screen_height / 2 else -0.85
            target_y -= contact_offset * self.rect.height / 2
        return target_y

    def update(self, ball_group) -> None:
        """Move toward a wall-reflected intercept instead of chasing the ball."""
        previous_y = self.rect.y
        if self.manual_control:
            self.rect.y += self.movement
        else:
            ball = ball_group.sprite
            difference = self._target_y(ball) - self.rect.centery
            if abs(difference) > self.dead_zone:
                self.rect.y += round(
                    max(-self.speed, min(self.speed, difference))
                )
        self.screen_constrain()
        self.velocity = self.rect.y - previous_y
