from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import deltacamera
from poseviz.gl.transforms import camera_to_gl_mvp, camera_to_gl_view


@dataclass
class CameraState:
    """Snapshot of camera state for history navigation.

    Either snapped to a display camera (snapped_to is set) or free position.
    """

    snapped_to: Optional[int] = None  # If set, re-snap to this camera index
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    azimuth: float = 0.0
    elevation: float = 0.0
    distance: float = 0.0
    fov: float = 55.0

    @property
    def is_snapped(self) -> bool:
        return self.snapped_to is not None


class TerrainCamera:
    """Orbit/fly camera that rotates around a pivot point.

    Supports:
    - Orbit: rotate around pivot (azimuth/elevation)
    - Pan: move pivot in view plane
    - Zoom: change distance from pivot
    - Fly: move pivot in camera direction (WASD-style)
    """

    def __init__(self, flying_mode: str = "camera"):
        """Initialize terrain camera.

        Args:
            flying_mode: "camera" for true 3D flying, "horizontal" for horizontal-only movement
        """
        self.pivot = np.array([0.0, 0.0, 2500.0], dtype=np.float32)
        self.azimuth = 0.0  # Horizontal angle around Y axis
        self.elevation = 0.0  # Vertical angle (0 = horizontal, positive = looking down)
        self.distance = 2500.0  # Distance from pivot
        self.fov = 55.0  # Field of view in degrees
        self.flying_mode = flying_mode
        self.initialized = False

        # Snap state: if set, we follow this display camera
        self.snapped_to: Optional[int] = None

        # Drag state (for "remember from start" interaction)
        self._drag_start_azimuth = 0.0
        self._drag_start_elevation = 0.0
        self._drag_start_pivot = None
        self._drag_start_distance = 2500.0

        # History for back/forward navigation
        self._back_history = []
        self._forward_history = []
        self._max_history = 50
        self._pending_checkpoint = None  # Snapped position waiting to be committed

    def init_from_camera(self, cam):
        """Initialize from a deltacamera.Camera, placing pivot in front of it.

        Args:
            cam: deltacamera.Camera object
        """
        # Pivot is some distance in front of camera (where skeleton likely is)
        self.pivot = cam.t + cam.R[2] * 2500
        self.distance = 2500.0

        # Extract azimuth and elevation from camera forward direction
        forward = cam.R[2]
        self.azimuth = np.arctan2(-forward[0], -forward[2])
        self.elevation = np.arcsin(np.clip(forward[1], -1, 1))
        self.initialized = True

    def get_position(self) -> np.ndarray:
        """Get camera position in world coordinates."""
        cos_el = np.cos(self.elevation)
        sin_el = np.sin(self.elevation)
        cos_az = np.cos(self.azimuth)
        sin_az = np.sin(self.azimuth)

        cam_x = self.pivot[0] + self.distance * sin_az * cos_el
        cam_y = self.pivot[1] - self.distance * sin_el  # Y is down, so negate
        cam_z = self.pivot[2] + self.distance * cos_az * cos_el
        return np.array([cam_x, cam_y, cam_z], dtype=np.float32)

    def get_view_proj(self, imshape: tuple = None) -> np.ndarray:
        """Get view-projection matrix.

        Args:
            imshape: (height, width) for aspect ratio. If None, uses 16:9.

        Returns:
            4x4 view-projection matrix (column-major for OpenGL)
        """
        cam_pos = self.get_position()

        if imshape is None:
            imshape = (720, 1280)  # Default 16:9

        # Create camera and use turn_towards for proper orientation
        # world_up for deltacamera: Y-down means up is (0, -1, 0)
        cam = (
            deltacamera.Camera.from_fov(self.fov, imshape, world_up=(0, -1, 0))
            .copy(optical_center=cam_pos)
            .turned_towards(target_world_point=self.pivot)
        )

        return camera_to_gl_mvp(cam, imshape)

    def get_view(self, imshape: tuple = None) -> np.ndarray:
        """Get view matrix (world-to-camera).

        Args:
            imshape: (height, width) for aspect ratio. If None, uses 16:9.

        Returns:
            4x4 view matrix (column-major for OpenGL)
        """
        cam_pos = self.get_position()

        if imshape is None:
            imshape = (720, 1280)

        cam = (
            deltacamera.Camera.from_fov(self.fov, imshape, world_up=(0, -1, 0))
            .copy(optical_center=cam_pos)
            .turned_towards(target_world_point=self.pivot)
        )

        return camera_to_gl_view(cam)

    # --- Drag interaction (remember from start) ---

    def begin_drag(self):
        """Call when starting a mouse drag. Saves current state."""
        self._commit_pending_checkpoint()
        self._drag_start_azimuth = self.azimuth
        self._drag_start_elevation = self.elevation
        self._drag_start_pivot = self.pivot.copy()
        self._drag_start_distance = self.distance
        self._drag_start_position = self.get_position()

    def orbit(self, dx: float, dy: float, sensitivity: float = 0.005):
        """Orbit camera around pivot based on mouse delta from drag start.

        Args:
            dx, dy: Mouse delta from drag start position
            sensitivity: Radians per pixel
        """
        self.azimuth = self._drag_start_azimuth + dx * sensitivity
        self.elevation = self._drag_start_elevation + dy * sensitivity
        # Clamp elevation to avoid gimbal lock
        self.elevation = np.clip(self.elevation, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)

    def look_around(self, dx: float, dy: float, sensitivity: float = 0.005):
        """Rotate viewing direction in place (camera position stays fixed).

        Like orbit, but the pivot moves around the camera instead of the camera
        moving around the pivot.

        Args:
            dx, dy: Mouse delta from drag start position
            sensitivity: Radians per pixel
        """
        self.azimuth = self._drag_start_azimuth + dx * sensitivity
        self.elevation = self._drag_start_elevation + dy * sensitivity
        self.elevation = np.clip(self.elevation, -np.pi / 2 + 0.1, np.pi / 2 - 0.1)

        # Recompute pivot so camera stays at its original position
        cos_el = np.cos(self.elevation)
        sin_el = np.sin(self.elevation)
        cos_az = np.cos(self.azimuth)
        sin_az = np.sin(self.azimuth)
        offset = self.distance * np.array(
            [sin_az * cos_el, -sin_el, cos_az * cos_el], dtype=np.float32
        )
        self.pivot = self._drag_start_position - offset

    def pan(self, dx: float, dy: float, sensitivity: float = None):
        """Pan camera (move pivot) based on mouse delta from drag start.

        Args:
            dx, dy: Mouse delta from drag start position
            sensitivity: Units per pixel. If None, auto-scales with distance.
        """
        if sensitivity is None:
            sensitivity = self.distance * 0.002

        # Right vector (perpendicular to view direction, horizontal)
        right = np.array(
            [np.cos(self._drag_start_azimuth), 0, -np.sin(self._drag_start_azimuth)]
        )
        # Up vector in world (Y is down, so negative Y is up)
        up = np.array([0, -1, 0])

        self.pivot = (
            self._drag_start_pivot + right * dx * sensitivity + up * dy * sensitivity
        )

    def zoom_drag(self, dy: float, sensitivity: float = 0.005):
        """Zoom based on vertical mouse delta from drag start.

        Args:
            dy: Vertical mouse delta from drag start
            sensitivity: Exponential factor per pixel
        """
        factor = np.exp(dy * sensitivity)
        self.distance = np.clip(self._drag_start_distance * factor, 100, 50000)

    def zoom_scroll(self, direction: float):
        """Zoom based on scroll wheel.

        Args:
            direction: Positive for zoom in, negative for zoom out
        """
        self._commit_pending_checkpoint()
        factor = 0.9 if direction > 0 else 1.1
        self.distance = np.clip(self.distance * factor, 100, 50000)

    def adjust_fov(self, delta: float):
        """Adjust field of view.

        Args:
            delta: Degrees to add (negative = zoom in, positive = zoom out)
        """
        self._commit_pending_checkpoint()
        self.fov = np.clip(self.fov + delta, 10.0, 120.0)

    # --- Flying controls ---

    def fly(self, direction: str, speed: float = None):
        """Move camera in a direction (WASD-style).

        Args:
            direction: One of "forward", "backward", "left", "right", "up", "down"
            speed: Movement amount. If None, auto-scales with distance.
        """
        self._commit_pending_checkpoint()
        if speed is None:
            speed = self.distance * 0.05

        cos_az = np.cos(self.azimuth)
        sin_az = np.sin(self.azimuth)

        if self.flying_mode == "camera":
            # Camera-relative: forward includes elevation
            cos_el = np.cos(self.elevation)
            sin_el = np.sin(self.elevation)

            # Forward direction toward pivot
            forward = np.array(
                [
                    -sin_az * cos_el,
                    sin_el,
                    -cos_az * cos_el,
                ],
                dtype=np.float32,
            )

            # Right direction (perpendicular, horizontal)
            right = np.array([-cos_az, 0, sin_az], dtype=np.float32)

            # Up direction (camera's local up)
            # right cross forward is always up in all right-handed systems
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)
        else:
            # Horizontal mode: forward stays horizontal
            forward = np.array([-sin_az, 0, -cos_az], dtype=np.float32)
            right = np.array([-cos_az, 0, sin_az], dtype=np.float32)
            up = np.array([0, -1, 0], dtype=np.float32)  # Y-down world

        if direction == "forward":
            self.pivot += forward * speed
        elif direction == "backward":
            self.pivot -= forward * speed
        elif direction == "left":
            self.pivot -= right * speed
        elif direction == "right":
            self.pivot += right * speed
        elif direction == "up":
            self.pivot += up * speed
        elif direction == "down":
            self.pivot -= up * speed

    # --- History navigation ---

    def get_state(self) -> CameraState:
        """Get current camera state as a snapshot."""
        return CameraState(
            snapped_to=self.snapped_to,
            pivot=self.pivot.copy(),
            azimuth=self.azimuth,
            elevation=self.elevation,
            distance=self.distance,
            fov=self.fov,
        )

    def set_state(self, state: CameraState):
        """Restore camera from a state snapshot."""
        self.snapped_to = state.snapped_to
        if not state.is_snapped:
            # Only restore position if not snapped (snapped state will be updated by renderer)
            self.pivot = state.pivot.copy()
            self.azimuth = state.azimuth
            self.elevation = state.elevation
            self.distance = state.distance
            self.fov = state.fov

    def snap_to(self, camera_index: int):
        """Snap to follow a display camera."""
        self.snapped_to = camera_index

    def unsnap(self):
        """Stop following display camera, keep current position."""
        self.snapped_to = None

    @property
    def is_snapped(self) -> bool:
        return self.snapped_to is not None

    def push_state(self):
        """Push current state to back history (call before jumping to a new view)."""
        self._back_history.append(self.get_state())
        if len(self._back_history) > self._max_history:
            self._back_history.pop(0)
        # Clear forward history when making a new move
        self._forward_history.clear()
        # Clear any pending checkpoint since we're making a new jump
        self._pending_checkpoint = None

    def set_pending_checkpoint(self):
        """Set current position as pending checkpoint (pushed to history on first movement)."""
        self._pending_checkpoint = self.get_state()

    def _commit_pending_checkpoint(self):
        """Push pending checkpoint to history and unsnap (called when user starts moving)."""
        if self._pending_checkpoint is not None:
            # Push the checkpoint position to history
            self._back_history.append(self._pending_checkpoint)
            if len(self._back_history) > self._max_history:
                self._back_history.pop(0)
            self._forward_history.clear()
            self._pending_checkpoint = None
        # User movement always unsnaps
        self.snapped_to = None

    def can_go_back(self) -> bool:
        """Check if back navigation is available."""
        return len(self._back_history) > 0

    def can_go_forward(self) -> bool:
        """Check if forward navigation is available."""
        return len(self._forward_history) > 0

    def go_back(self) -> bool:
        """Navigate to previous camera position.

        Returns:
            True if navigation occurred, False if no history.
        """
        if not self._back_history:
            return False
        # Save current state to forward history
        self._forward_history.append(self.get_state())
        # Restore previous state
        state = self._back_history.pop()
        self.set_state(state)
        return True

    def go_forward(self) -> bool:
        """Navigate to next camera position (after going back).

        Returns:
            True if navigation occurred, False if no forward history.
        """
        if not self._forward_history:
            return False
        # Save current state to back history
        self._back_history.append(self.get_state())
        # Restore forward state
        state = self._forward_history.pop()
        self.set_state(state)
        return True
