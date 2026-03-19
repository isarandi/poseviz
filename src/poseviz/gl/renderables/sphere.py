import numpy as np
import moderngl

from poseviz.gl.renderables.base import ShaderRenderable
from poseviz.gl.renderables.mixins import InstancedMixin


class SphereRenderable(ShaderRenderable, InstancedMixin):
    """Renders many spheres via instancing."""

    def __init__(
        self,
        ctx: moderngl.Context,
        color: tuple,
        resolution: int = 16,
        max_instances: int = 1000,
    ):
        super().__init__(ctx)
        self.color = color
        self._load_program("sphere")

        # Create unit sphere geometry
        vertices, normals, indices = _generate_sphere(resolution)

        vbo_vertices = ctx.buffer(vertices.tobytes())
        vbo_normals = ctx.buffer(normals.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        self._register_resources(vbo_vertices, vbo_normals, ibo)

        # Instance buffer: [x, y, z, scale] per instance
        self._init_instance_buffer(max_instances, stride=16)
        self._register_resources(self.instance_vbo)

        self.vao = ctx.vertex_array(
            self.program,
            [
                (vbo_vertices, "3f", "in_position"),
                (vbo_normals, "3f", "in_normal"),
                (self.instance_vbo, "3f 1f /i", "instance_pos", "instance_scale"),
            ],
            ibo,
        )

    def update(self, positions: np.ndarray, scale: float = 0.06):
        """Update sphere positions.

        Args:
            positions: (N, 3) array of positions
            scale: Sphere radius
        """
        if len(positions) == 0:
            self.instance_count = 0
            return

        n = min(len(positions), self.max_instances)
        data = np.zeros((n, 4), dtype=np.float32)
        data[:, :3] = positions[:n]
        data[:, 3] = scale
        self._update_instances(data)

    def render(self, view_proj: np.ndarray):
        if self.instance_count == 0:
            return

        self._set_view_proj(view_proj)
        self._set_color()
        self.vao.render(moderngl.TRIANGLES, instances=self.instance_count)


def _generate_sphere(resolution: int) -> tuple:
    """Generate unit sphere vertices, normals, and indices."""
    vertices = []
    normals = []

    for i in range(resolution + 1):
        lat = np.pi * i / resolution - np.pi / 2
        for j in range(resolution + 1):
            lon = 2 * np.pi * j / resolution
            x = np.cos(lat) * np.cos(lon)
            y = np.sin(lat)
            z = np.cos(lat) * np.sin(lon)
            vertices.append([x, y, z])
            normals.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    indices = []
    for i in range(resolution):
        for j in range(resolution):
            p0 = i * (resolution + 1) + j
            p1 = p0 + 1
            p2 = p0 + resolution + 1
            p3 = p2 + 1
            indices.extend([p0, p2, p1, p1, p2, p3])

    return vertices, normals, np.array(indices, dtype=np.uint32)
