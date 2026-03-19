from abc import abstractmethod
import numpy as np
import moderngl

from poseviz.gl.renderable import Renderable
from poseviz.gl.shader_loader import load_program


class ShaderRenderable(Renderable):
    """Base renderable with shader program, VAO, and automatic resource cleanup.

    Subclasses should:
    - Call _register_resources() with all created GPU resources
    - Optionally set self.color for uniform color support
    """

    _current_view: bytes = None  # Set by renderer before each render pass

    def __init__(self, ctx: moderngl.Context):
        super().__init__(ctx)
        self.program: moderngl.Program = None
        self.vao: moderngl.VertexArray = None
        self.color: tuple = None
        self._resources: list = []

    def _load_program(self, name: str):
        """Load shader program by name."""
        self.program = load_program(self.ctx, name)
        self._resources.append(self.program)

    def _register_resources(self, *resources):
        """Register GPU resources for automatic cleanup."""
        self._resources.extend(resources)

    def _set_view_proj(self, view_proj: np.ndarray):
        """Set view-projection matrix uniform (and view matrix if shader uses it)."""
        self.program["u_view_proj"].write(view_proj.astype(np.float32).tobytes())
        if self._current_view is not None and "u_view" in self.program:
            self.program["u_view"].write(self._current_view)

    def _set_color(self):
        """Set uniform color if defined."""
        if self.color is not None:
            self.program["u_color"].value = self.color

    @abstractmethod
    def render(self, view_proj: np.ndarray):
        pass

    def destroy(self):
        """Release all registered GPU resources."""
        if self.vao:
            self.vao.release()
        for resource in self._resources:
            resource.release()
        self._resources.clear()
