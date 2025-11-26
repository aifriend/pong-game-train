"""Point class for vector math operations."""

import math
from typing import Tuple


class Point:
    """2D point/vector class with basic math operations."""

    def __init__(self, point_t: Tuple[float, float] = (0, 0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])

    def __add__(self, other: "Point") -> "Point":
        """Add two points."""
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other: "Point") -> "Point":
        """Subtract two points."""
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar: float) -> "Point":
        """Multiply point by scalar."""
        return Point((self.x * scalar, self.y * scalar))

    def __truediv__(self, scalar: float) -> "Point":
        """Divide point by scalar."""
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        return Point((self.x / scalar, self.y / scalar))

    def __len__(self) -> int:
        """Return the magnitude (length) of the vector."""
        return int(math.sqrt(self.x**2 + self.y**2))

    def get(self) -> Tuple[float, float]:
        """Return point as tuple."""
        return self.x, self.y
