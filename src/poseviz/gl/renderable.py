from abc import ABC, abstractmethod
import moderngl


class Renderable(ABC):
    """Base class for all GPU-rendered primitives."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx

    @abstractmethod
    def render(self, view_proj: bytes):
        """Render the primitive. view_proj is the combined view-projection matrix as bytes."""
        pass

    def destroy(self):
        """Release GPU resources. Override if needed."""
        pass
