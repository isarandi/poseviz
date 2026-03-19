import numpy as np
import moderngl

from poseviz.gl.renderables.colormap import Colormap


class TextureMixin:
    """Mixin for renderables that use textures."""

    texture: moderngl.Texture = None
    _texture_shape: tuple = None
    _cuda_writer = None

    def _create_texture(self, width: int, height: int, components: int = 3):
        """Create or recreate texture with given dimensions."""
        if self.texture is not None:
            self.texture.release()
        self.texture = self.ctx.texture((width, height), components)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._texture_shape = (height, width)

    def _update_texture(self, image):
        """Upload image to texture, creating/resizing as needed.

        Accepts numpy arrays (CPU upload) or CUDA tensors (GPU-internal copy
        via CUDA-GL interop).
        """
        h, w = image.shape[:2]
        components = image.shape[2] if image.ndim > 2 else 1
        is_cuda = not isinstance(image, np.ndarray)

        if is_cuda and components == 3:
            # CUDA-GL interop requires RGBA — GL_RGB8 is stored as RGBA internally,
            # so cudaMemcpy2DToArray with 3 bytes/pixel causes stride mismatch.
            import torch
            alpha = torch.full((h, w, 1), 255, dtype=image.dtype, device=image.device)
            image = torch.cat([image, alpha], dim=2).contiguous()
            components = 4

        if self.texture is None or (h, w) != self._texture_shape or self.texture.components != components:
            self._create_texture(w, h, components)

        if not is_cuda:
            self.texture.write(image)
        else:
            if self._cuda_writer is None:
                from poseviz.gl.cuda_gl_interop import CudaGLTextureWriter
                self._cuda_writer = CudaGLTextureWriter()
            self._cuda_writer.write(image, self.texture)

    def _bind_texture(self, slot: int = 0, uniform: str = "u_texture"):
        """Bind texture and set sampler uniform."""
        if self.texture:
            self.texture.use(slot)
            self.program[uniform].value = slot

    def _release_texture(self):
        """Release texture."""
        if self._cuda_writer is not None:
            self._cuda_writer.release()
            self._cuda_writer = None
        if self.texture:
            self.texture.release()
            self.texture = None


class InstancedMixin:
    """Mixin for instanced rendering."""

    instance_vbo: moderngl.Buffer = None
    instance_count: int = 0
    max_instances: int = 0

    def _init_instance_buffer(self, max_instances: int, stride: int):
        """Create instance buffer.

        Args:
            max_instances: Maximum number of instances
            stride: Bytes per instance
        """
        self.max_instances = max_instances
        self.instance_vbo = self.ctx.buffer(reserve=max_instances * stride)

    def _update_instances(self, data: np.ndarray) -> int:
        """Update instance buffer.

        Args:
            data: (N, ...) array of instance data

        Returns:
            Actual instance count (clamped to max)
        """
        if len(data) == 0:
            self.instance_count = 0
            return 0
        n = min(len(data), self.max_instances)
        self.instance_vbo.write(data[:n].astype(np.float32).tobytes())
        self.instance_count = n
        return n

    def _release_instance_buffer(self):
        """Release instance buffer."""
        if self.instance_vbo:
            self.instance_vbo.release()
            self.instance_vbo = None


class ScalarColormapMixin:
    """Mixin for per-vertex scalar visualization via colormap."""

    scalar_vbo: moderngl.Buffer = None
    colormap: Colormap = None
    vmin: float = 0.0
    vmax: float = 1.0

    def _init_scalar_colormap(self, max_vertices: int, colormap_name: str = "viridis"):
        """Initialize scalar buffer and colormap."""
        self.scalar_vbo = self.ctx.buffer(reserve=max_vertices * 4)
        self.colormap = Colormap.get(self.ctx, colormap_name)

    def _update_scalars(self, scalars: np.ndarray, auto_range: bool = False):
        """Update per-vertex scalars.

        Args:
            scalars: (N,) array of scalar values
            auto_range: If True, set vmin/vmax from data
        """
        scalars = scalars.astype(np.float32)
        if auto_range and len(scalars) > 0:
            self.vmin = float(scalars.min())
            self.vmax = float(scalars.max())
        self.scalar_vbo.write(scalars.tobytes())

    def set_scalar_range(self, vmin: float, vmax: float):
        """Set scalar normalization range."""
        self.vmin = vmin
        self.vmax = vmax

    def set_colormap(self, name: str):
        """Change colormap."""
        self.colormap = Colormap.get(self.ctx, name)

    def _bind_scalar_colormap(self, slot: int = 1):
        """Bind colormap and set uniforms."""
        self.colormap.bind(slot)
        self.program["u_colormap"].value = slot
        self.program["u_vmin"].value = self.vmin
        self.program["u_vmax"].value = self.vmax

    def _release_scalar_colormap(self):
        """Release scalar buffer (colormap is cached, not released)."""
        if self.scalar_vbo:
            self.scalar_vbo.release()
            self.scalar_vbo = None
