import numpy as np
from poseviz.gl.transforms import world_to_gl, WORLD_TO_GL_ROTATION_MAT


class Camera:
    """Terrain-style camera that orbits around a focal point."""

    def __init__(self):
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.focal_point = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.fov = 45.0
        self.near = 0.1
        self.far = 1000.0
        self.aspect = 16 / 9

    def set_from_world_camera(
        self, world_camera, image_size=None, camera_view_padding=0.2
    ):
        """Set view from a world-space camera (deltacamera.Camera)."""
        # Focal point: some distance in front of camera
        pivot = world_camera.t + world_camera.R[2] * 2500
        self.focal_point = world_to_gl(pivot)
        self.position = world_to_gl(world_camera.t)
        self.up = -WORLD_TO_GL_ROTATION_MAT @ world_camera.R[1]

        if image_size is not None:
            half_height = 0.5 * image_size[1]
            offset = np.abs(world_camera.intrinsic_matrix[1, 2] - half_height)
            tan = (
                (1 + camera_view_padding)
                * (half_height + offset)
                / world_camera.intrinsic_matrix[1, 1]
            )
            self.fov = np.rad2deg(2 * np.arctan(tan))

    def rotate(self, delta_azimuth: float, delta_elevation: float):
        """Rotate camera around focal point (terrain style).

        Args:
            delta_azimuth: Horizontal rotation in radians
            delta_elevation: Vertical rotation in radians
        """
        # Vector from focal point to camera
        offset = self.position - self.focal_point
        distance = np.linalg.norm(offset)

        # Current spherical coordinates
        azimuth = np.arctan2(offset[0], offset[2])
        elevation = np.arcsin(np.clip(offset[1] / distance, -1, 1))

        # Apply deltas
        azimuth += delta_azimuth
        elevation = np.clip(
            elevation + delta_elevation, -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        )

        # Convert back to cartesian
        self.position = self.focal_point + distance * np.array(
            [
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
                np.cos(elevation) * np.cos(azimuth),
            ],
            dtype=np.float32,
        )

    def dolly(self, delta: float):
        """Move camera toward/away from focal point."""
        offset = self.position - self.focal_point
        distance = np.linalg.norm(offset)
        new_distance = max(0.1, distance * (1.0 - delta))
        self.position = self.focal_point + (offset / distance) * new_distance

    def get_view_matrix(self) -> np.ndarray:
        """Get 4x4 view matrix."""
        f = _normalize(self.focal_point - self.position)
        r = _normalize(np.cross(f, self.up))
        u = np.cross(r, f)

        return np.array(
            [
                [r[0], r[1], r[2], -np.dot(r, self.position)],
                [u[0], u[1], u[2], -np.dot(u, self.position)],
                [-f[0], -f[1], -f[2], np.dot(f, self.position)],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

    def get_projection_matrix(self) -> np.ndarray:
        """Get 4x4 perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(self.fov) / 2)
        return np.array(
            [
                [f / self.aspect, 0, 0, 0],
                [0, f, 0, 0],
                [
                    0,
                    0,
                    (self.far + self.near) / (self.near - self.far),
                    2 * self.far * self.near / (self.near - self.far),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    def get_view_projection_matrix(self) -> np.ndarray:
        """Get combined view-projection matrix (column-major for OpenGL)."""
        vp = self.get_projection_matrix() @ self.get_view_matrix()
        return np.asfortranarray(vp)


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
