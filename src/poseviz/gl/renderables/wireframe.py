import numpy as np
import moderngl

from poseviz.gl.renderables.base import ShaderRenderable


class WireframeRenderable(ShaderRenderable):
    """Renders wireframe lines (camera pyramid, FOV cone edges)."""

    def __init__(self, ctx: moderngl.Context, color: tuple, max_vertices: int = 100):
        super().__init__(ctx)
        self.color = color
        self.max_vertices = max_vertices
        self.vertex_count = 0

        self._load_program("line")

        vbo = ctx.buffer(reserve=max_vertices * 12)
        self._register_resources(vbo)
        self._vbo = vbo

        self.vao = ctx.vertex_array(
            self.program,
            [(vbo, "3f", "in_position")],
        )

    def update_pyramid(self, camera, image_shape, image_plane_distance: float = 1000):
        """Update to show camera pyramid wireframe.

        Args:
            camera: deltacamera.Camera object
            image_shape: (H, W, ...) image shape
            image_plane_distance: Distance from camera to image corners
        """
        h, w = image_shape[:2]
        corners_2d = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        corners_world = camera.image_to_world(
            corners_2d, camera_depth=image_plane_distance
        )
        apex = camera.t

        # Lines: apex to each corner + rectangle
        lines = []
        for corner in corners_world:
            lines.extend([apex, corner])
        for i in range(4):
            lines.extend([corners_world[i], corners_world[(i + 1) % 4]])

        vertices = np.array(lines, dtype=np.float32)
        self._vbo.write(vertices.tobytes())
        self.vertex_count = len(vertices)

    def update_lines(self, vertices: np.ndarray):
        """Update with arbitrary line segments.

        Args:
            vertices: (N*2, 3) array of line endpoints
        """
        if len(vertices) == 0:
            self.vertex_count = 0
            return

        vertices = vertices.astype(np.float32)
        self._vbo.write(vertices.tobytes())
        self.vertex_count = len(vertices)

    def render(self, view_proj: np.ndarray):
        if self.vertex_count == 0:
            return

        self._set_view_proj(view_proj)
        self._set_color()
        self.vao.render(moderngl.LINES, vertices=self.vertex_count)
