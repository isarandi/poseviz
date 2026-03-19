import numpy as np
import moderngl
from poseviz.gl.shader_loader import load_program


class PyramidPicker:
    """Renders camera pyramids as solid geometry for color-based picking."""

    def __init__(self, ctx: moderngl.Context, resolution: tuple):
        self.ctx = ctx
        self.resolution = resolution  # (width, height)

        # Create offscreen framebuffer for picking
        self.color_attachment = ctx.texture(resolution, 4, dtype="f1")
        self.depth_attachment = ctx.depth_texture(resolution)
        self.fbo = ctx.framebuffer(
            color_attachments=[self.color_attachment],
            depth_attachment=self.depth_attachment,
        )

        # Shader - reuse line shader (flat color)
        self.program = load_program(ctx, "line")

        # VBO for pyramid triangles (will be updated dynamically)
        # Each pyramid: 4 side triangles + 2 base triangles = 6 triangles = 18 vertices
        self.max_cameras = 32
        self.vbo = ctx.buffer(
            reserve=self.max_cameras * 18 * 3 * 4
        )  # 18 verts * 3 floats * 4 bytes

        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, "3f", "in_position")],
        )

        # Store pyramid data per camera
        self.pyramid_data = []  # List of (apex, corners) tuples

    def update_camera(
        self, camera_index: int, camera, image_shape, image_plane_distance: float
    ):
        """Update pyramid geometry for a camera."""
        h, w = image_shape[:2]

        # Image corners in 2D
        corners_2d = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )

        corners_world = camera.image_to_world(
            corners_2d, camera_depth=image_plane_distance
        )
        apex = camera.t.astype(np.float32)

        # Ensure we have enough slots
        while len(self.pyramid_data) <= camera_index:
            self.pyramid_data.append(None)

        self.pyramid_data[camera_index] = (apex, corners_world)

    def clear_cameras(self):
        """Clear all camera pyramid data."""
        self.pyramid_data = []

    def resize(self, width: int, height: int):
        """Resize the picking framebuffer."""
        self.resolution = (width, height)
        self.color_attachment.release()
        self.depth_attachment.release()
        self.fbo.release()

        self.color_attachment = self.ctx.texture(self.resolution, 4, dtype="f1")
        self.depth_attachment = self.ctx.depth_texture(self.resolution)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_attachment],
            depth_attachment=self.depth_attachment,
        )

    def pick(self, x: int, y: int, view_proj: np.ndarray) -> int:
        """Render to picking buffer and return camera index at (x, y).

        Args:
            x, y: Screen coordinates (origin at top-left)
            view_proj: View-projection matrix

        Returns:
            Camera index (0-based) or -1 if no camera hit
        """
        # Render to picking framebuffer
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.program["u_view_proj"].write(view_proj.astype(np.float32).tobytes())

        # Render each camera pyramid with unique color
        for cam_idx, data in enumerate(self.pyramid_data):
            if data is None:
                continue

            apex, corners = data
            triangles = self._build_pyramid_triangles(apex, corners)

            self.vbo.write(triangles.tobytes())

            # Encode camera index as color (index + 1 to distinguish from background)
            # R channel = (index + 1) / 255
            color_id = (cam_idx + 1) / 255.0
            self.program["u_color"].value = (color_id, 0.0, 0.0)

            self.vao.render(moderngl.TRIANGLES, vertices=len(triangles))

        # Read pixel at click position
        # Note: OpenGL origin is bottom-left, so flip Y
        gl_y = self.resolution[1] - 1 - y
        pixel_bytes = self.fbo.read(viewport=(x, gl_y, 1, 1), components=4, dtype="f1")

        # pixel_bytes is a bytes object, each byte is a value 0-255
        r = pixel_bytes[0]

        # Decode camera index (we encoded as (index + 1) / 255.0, so r = index + 1)
        cam_idx = r - 1

        # Restore default framebuffer
        self.ctx.screen.use()

        return cam_idx if 0 <= cam_idx < len(self.pyramid_data) else -1

    def _build_pyramid_triangles(
        self, apex: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """Build triangle vertices for a pyramid.

        Args:
            apex: Camera position (3,)
            corners: Image plane corners (4, 3) in order: TL, TR, BR, BL

        Returns:
            (N, 3) array of triangle vertices
        """
        triangles = []

        # 4 side triangles (apex to each edge of rectangle)
        for i in range(4):
            triangles.append(apex)
            triangles.append(corners[i])
            triangles.append(corners[(i + 1) % 4])

        # 2 triangles for the base (image plane)
        triangles.append(corners[0])
        triangles.append(corners[1])
        triangles.append(corners[2])

        triangles.append(corners[0])
        triangles.append(corners[2])
        triangles.append(corners[3])

        return np.array(triangles, dtype=np.float32)

    def destroy(self):
        """Release GPU resources."""
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.fbo.release()
        self.color_attachment.release()
        self.depth_attachment.release()
