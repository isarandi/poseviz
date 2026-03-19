import numpy as np
import moderngl

from poseviz.gl.renderables.base import ShaderRenderable
from poseviz.gl.renderables.mixins import InstancedMixin


class TubeRenderable(ShaderRenderable, InstancedMixin):
    """Renders many tubes/cylinders via instancing."""

    def __init__(
        self,
        ctx: moderngl.Context,
        color: tuple,
        sides: int = 12,
        max_instances: int = 1000,
    ):
        super().__init__(ctx)
        self.color = color
        self._load_program("tube")

        # Create unit cylinder
        vertices, normals, indices = _generate_cylinder(sides)

        vbo_vertices = ctx.buffer(vertices.tobytes())
        vbo_normals = ctx.buffer(normals.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        self._register_resources(vbo_vertices, vbo_normals, ibo)

        # Instance buffer: [start(3), end(3), radius(1)] = 28 bytes
        self._init_instance_buffer(max_instances, stride=28)
        self._register_resources(self.instance_vbo)

        self.vao = ctx.vertex_array(
            self.program,
            [
                (vbo_vertices, "3f", "in_position"),
                (vbo_normals, "3f", "in_normal"),
                (
                    self.instance_vbo,
                    "3f 3f 1f /i",
                    "instance_start",
                    "instance_end",
                    "instance_radius",
                ),
            ],
            ibo,
        )

    def update(self, starts: np.ndarray, ends: np.ndarray, radius: float = 0.012):
        """Update tube endpoints.

        Args:
            starts: (N, 3) array of start positions
            ends: (N, 3) array of end positions
            radius: Tube radius
        """
        if len(starts) == 0:
            self.instance_count = 0
            return

        n = min(len(starts), self.max_instances)
        data = np.zeros((n, 7), dtype=np.float32)
        data[:, :3] = starts[:n]
        data[:, 3:6] = ends[:n]
        data[:, 6] = radius
        self._update_instances(data)

    def render(self, view_proj: np.ndarray):
        if self.instance_count == 0:
            return

        self._set_view_proj(view_proj)
        self._set_color()
        self.vao.render(moderngl.TRIANGLES, instances=self.instance_count)


def _generate_cylinder(sides: int) -> tuple:
    """Generate unit cylinder vertices, normals, and indices.

    Cylinder is along Y axis, radius 1, from y=0 to y=1.
    """
    vertices = []
    normals = []

    for y in [0.0, 1.0]:
        for i in range(sides):
            angle = 2 * np.pi * i / sides
            x = np.cos(angle)
            z = np.sin(angle)
            vertices.append([x, y, z])
            normals.append([x, 0, z])

    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)

    indices = []
    for i in range(sides):
        i0 = i
        i1 = (i + 1) % sides
        i2 = i + sides
        i3 = i1 + sides
        indices.extend([i0, i2, i1, i1, i2, i3])

    return vertices, normals, np.array(indices, dtype=np.uint32)
