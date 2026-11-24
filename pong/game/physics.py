"""Shared, frame-independent Pong trajectory and rebound calculations."""

import math
from typing import Tuple


def reflect_position(value: float, minimum: float, maximum: float) -> float:
    """Reflect a position between two walls without iterative clamping."""
    if minimum >= maximum:
        return minimum
    period = 2.0 * (maximum - minimum)
    reflected = (value - minimum) % period
    if reflected > maximum - minimum:
        reflected = period - reflected
    return minimum + reflected


def predict_intercept(
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    target_x: float,
    min_y: float,
    max_y: float,
) -> Tuple[float, float]:
    """Return the wall-reflected y position and time at a vertical plane."""
    if abs(ball_vx) < 1e-9:
        return ball_y, math.inf
    time_to_plane = (target_x - ball_x) / ball_vx
    if time_to_plane < 0:
        return ball_y, math.inf
    unbounded_y = ball_y + ball_vy * time_to_plane
    return reflect_position(unbounded_y, min_y, max_y), time_to_plane


def bounce_velocity(
    ball_vx: float,
    ball_vy: float,
    hit_offset: float,
    direction: int,
    paddle_velocity: float,
    max_paddle_speed: float,
    acceleration: float,
    max_speed: float,
    max_bounce_angle: float,
    paddle_spin_angle: float,
) -> Tuple[float, float]:
    """Return a constant-speed rebound controlled by contact and paddle motion."""
    speed = min(
        math.hypot(ball_vx, ball_vy) * (1.0 + acceleration),
        max_speed,
    )
    offset = max(-1.0, min(1.0, hit_offset))
    spin = max(-1.0, min(1.0, paddle_velocity / max_paddle_speed))
    angle = offset * max_bounce_angle + spin * paddle_spin_angle
    angle = max(-75.0, min(75.0, angle))
    radians = math.radians(angle)
    return direction * speed * math.cos(radians), speed * math.sin(radians)
