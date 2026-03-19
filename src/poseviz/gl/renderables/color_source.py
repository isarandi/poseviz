from abc import ABC, abstractmethod
import numpy as np
import moderngl

from poseviz.gl.renderables.colormap import Colormap


class ColorSource(ABC):
    """Abstract color source for mesh rendering."""

    shader_name: str = None  # Shader to use with this color source

    @abstractmethod
    def setup(self, ctx: moderngl.Context, max_vertices: int):
        """Set up any GPU resources needed."""
        pass

    @abstractmethod
    def get_vao_content(self) -> list:
        """Return list of (buffer, format, *attributes) for VAO creation.

        Returns empty list if no additional vertex attributes needed.
        """
        pass

    @abstractmethod
    def update(self, data):
        """Update color data. Data format depends on subclass."""
        pass

    @abstractmethod
    def bind(self, program: moderngl.Program):
        """Bind uniforms/textures before rendering."""
        pass

    @abstractmethod
    def release(self):
        """Release GPU resources."""
        pass


class UniformColor(ColorSource):
    """Single color for entire mesh."""

    shader_name = "mesh"

    def __init__(self, color: tuple):
        self.color = color

    def setup(self, ctx: moderngl.Context, max_vertices: int):
        pass  # No GPU resources needed

    def get_vao_content(self) -> list:
        return []  # No additional vertex attributes

    def update(self, data):
        """Update color. data is a tuple (r, g, b)."""
        self.color = data

    def bind(self, program: moderngl.Program):
        program["u_color"].value = self.color

    def release(self):
        pass


class VertexRGBColor(ColorSource):
    """Per-vertex RGB colors."""

    shader_name = "mesh_vertexcolor"

    def __init__(self):
        self.vbo: moderngl.Buffer = None
        self.ctx: moderngl.Context = None

    def setup(self, ctx: moderngl.Context, max_vertices: int):
        self.ctx = ctx
        self.vbo = ctx.buffer(reserve=max_vertices * 12)  # 3 floats per vertex

    def get_vao_content(self) -> list:
        return [(self.vbo, "3f", "in_color")]

    def update(self, data: np.ndarray):
        """Update colors. data is (N, 3) array of RGB values (0-1)."""
        if data is not None and len(data) > 0:
            self.vbo.write(data.astype(np.float32).tobytes())

    def bind(self, program: moderngl.Program):
        pass  # Color comes from vertex attribute, no uniforms

    def release(self):
        if self.vbo:
            self.vbo.release()


class ScalarColormapColor(ColorSource):
    """Per-vertex scalar mapped through colormap."""

    shader_name = "mesh_scalar"

    def __init__(
        self, colormap_name: str = "viridis", vmin: float = 0.0, vmax: float = 1.0
    ):
        self.colormap_name = colormap_name
        self.vmin = vmin
        self.vmax = vmax
        self.vbo: moderngl.Buffer = None
        self.colormap: Colormap = None
        self.ctx: moderngl.Context = None

    def setup(self, ctx: moderngl.Context, max_vertices: int):
        self.ctx = ctx
        self.vbo = ctx.buffer(reserve=max_vertices * 4)  # 1 float per vertex
        self.colormap = Colormap.get(ctx, self.colormap_name)

    def get_vao_content(self) -> list:
        return [(self.vbo, "1f", "in_scalar")]

    def update(self, data: np.ndarray, auto_range: bool = False):
        """Update scalars. data is (N,) array of scalar values."""
        if data is not None and len(data) > 0:
            data = data.astype(np.float32)
            if auto_range:
                self.vmin = float(data.min())
                self.vmax = float(data.max())
            self.vbo.write(data.tobytes())

    def set_range(self, vmin: float, vmax: float):
        """Set scalar normalization range."""
        self.vmin = vmin
        self.vmax = vmax

    def set_colormap(self, name: str):
        """Change colormap."""
        self.colormap_name = name
        if self.ctx:
            self.colormap = Colormap.get(self.ctx, name)

    def bind(self, program: moderngl.Program):
        self.colormap.bind(slot=1)
        program["u_colormap"].value = 1
        program["u_vmin"].value = self.vmin
        program["u_vmax"].value = self.vmax

    def release(self):
        if self.vbo:
            self.vbo.release()
        # Don't release colormap - it's cached


class TextureColor(ColorSource):
    """UV-mapped texture."""

    shader_name = "mesh_textured"

    def __init__(self, opacity: float = 1.0):
        self.opacity = opacity
        self.texture: moderngl.Texture = None
        self.texcoord_vbo: moderngl.Buffer = None
        self.ctx: moderngl.Context = None
        self._texture_shape: tuple = None

    def setup(self, ctx: moderngl.Context, max_vertices: int):
        self.ctx = ctx
        self.texcoord_vbo = ctx.buffer(reserve=max_vertices * 8)

    def get_vao_content(self) -> list:
        return [(self.texcoord_vbo, "2f", "in_texcoord")]

    def set_texcoords(self, texcoords: np.ndarray):
        """Set UV coordinates (typically once per mesh)."""
        self.texcoord_vbo.write(texcoords.astype(np.float32).tobytes())

    def set_texture(self, image: np.ndarray):
        """Set texture image."""
        if image is None:
            return
        h, w = image.shape[:2]
        components = image.shape[2] if image.ndim > 2 else 1
        if self.texture is None or (h, w) != self._texture_shape:
            if self.texture is not None:
                self.texture.release()
            self.texture = self.ctx.texture((w, h), components)
            self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._texture_shape = (h, w)
        self.texture.write(np.ascontiguousarray(image).tobytes())

    def update(self, data):
        """Update texture image. Alias for set_texture for ColorSource interface."""
        self.set_texture(data)

    def bind(self, program: moderngl.Program):
        if self.texture:
            self.texture.use(0)
            program["u_texture"].value = 0
            program["u_opacity"].value = self.opacity

    def release(self):
        if self.texcoord_vbo:
            self.texcoord_vbo.release()
        if self.texture:
            self.texture.release()
