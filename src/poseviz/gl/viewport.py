from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Viewport:
    """A viewport defines a screen region with its own camera/projection.

    Attributes:
        name: Identifier for this viewport (e.g., "original", "terrain")
        bounds: Screen region as (x, y, width, height) in pixels
        get_view_proj: Callable that returns the view-projection matrix for this viewport
        interactive: Whether this viewport receives mouse/keyboard input for camera control
    """

    name: str
    bounds: tuple  # (x, y, width, height)
    get_view_proj: Callable[[], np.ndarray]
    get_matrices: Callable[[], tuple] = None  # Returns (view_proj, view)
    interactive: bool = False

    @property
    def x(self) -> int:
        return self.bounds[0]

    @property
    def y(self) -> int:
        return self.bounds[1]

    @property
    def width(self) -> int:
        return self.bounds[2]

    @property
    def height(self) -> int:
        return self.bounds[3]

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    def contains(self, screen_x: int, screen_y: int) -> bool:
        """Check if screen coordinates are inside this viewport."""
        return (
            self.x <= screen_x < self.x + self.width
            and self.y <= screen_y < self.y + self.height
        )

    def to_local(self, screen_x: int, screen_y: int) -> tuple:
        """Transform screen coordinates to viewport-local coordinates.

        Returns:
            (local_x, local_y) relative to viewport origin
        """
        return (screen_x - self.x, screen_y - self.y)

    def to_normalized(self, screen_x: int, screen_y: int) -> tuple:
        """Transform screen coordinates to normalized viewport coordinates [0, 1].

        Returns:
            (norm_x, norm_y) in range [0, 1]
        """
        local_x, local_y = self.to_local(screen_x, screen_y)
        return (local_x / self.width, local_y / self.height)
