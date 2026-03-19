import numpy as np
import moderngl

from .base import ShaderRenderable
from .mixins import TextureMixin


class ImageQuadRenderable(ShaderRenderable, TextureMixin):
    """Renders a textured quad positioned in 3D space (camera image plane)."""

    def __init__(self, ctx: moderngl.Context, opacity: float = 0.5):
        super().__init__(ctx)
        self.opacity = opacity
        self.visible = False
        self._load_program("textured")

        # Quad vertices: [x, y, z, u, v] * 4
        vbo = ctx.buffer(reserve=4 * 20)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        ibo = ctx.buffer(indices.tobytes())
        self._register_resources(vbo, ibo)
        self._vbo = vbo

        self.vao = ctx.vertex_array(
            self.program,
            [(vbo, "3f 2f", "in_position", "in_texcoord")],
            ibo,
        )

    def update(self, camera, image: np.ndarray, image_plane_distance: float = 1000):
        """Update image and quad position.

        Args:
            camera: deltacamera.Camera object
            image: (H, W, 3) RGB image
            image_plane_distance: Distance from camera to image plane in world units
        """
        if image is None:
            self.visible = False
            return

        h, w = image.shape[:2]

        self._update_texture(image)

        # Compute quad corners in world space
        image_corners = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        corners_world = camera.image_to_world(
            image_corners, camera_depth=image_plane_distance
        )

        # Build vertex data: [x, y, z, u, v]
        texcoords = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ],
            dtype=np.float32,
        )

        vertex_data = np.zeros((4, 5), dtype=np.float32)
        vertex_data[:, :3] = corners_world
        vertex_data[:, 3:5] = texcoords

        self._vbo.write(vertex_data.tobytes())
        self.visible = True

    def render(self, view_proj: np.ndarray):
        if not self.visible or self.texture is None:
            return

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.depth_mask = False

        self._bind_texture(0)
        self.program["u_opacity"].value = self.opacity
        self._set_view_proj(view_proj)

        self.vao.render(moderngl.TRIANGLES)

        self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

    def destroy(self):
        self._release_texture()
        super().destroy()
