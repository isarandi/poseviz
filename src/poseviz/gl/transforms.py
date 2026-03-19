import numpy as np

MM_TO_UNIT = 1 / 1000
UNIT_TO_MM = 1000

# Default world up is Y-down (common in pose estimation)
# This matrix rotates world coords to GL coords (Y-up, Z-out-of-screen)
WORLD_TO_GL_ROTATION_MAT = np.array(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
)


def set_world_up(world_up):
    """Set the world up vector, updates the rotation matrix accordingly."""
    global WORLD_TO_GL_ROTATION_MAT
    WORLD_TO_GL_ROTATION_MAT = _rotation_mat(world_up)


def world_to_gl(points):
    """Convert world coordinates to GL coordinates."""
    points = np.asarray(points)
    if points.ndim == 1:
        return WORLD_TO_GL_ROTATION_MAT @ points * MM_TO_UNIT
    else:
        return points @ WORLD_TO_GL_ROTATION_MAT.T * MM_TO_UNIT


def gl_to_world(points):
    """Convert GL coordinates to world coordinates."""
    points = np.asarray(points)
    if points.ndim == 1:
        return WORLD_TO_GL_ROTATION_MAT.T @ points * UNIT_TO_MM
    else:
        return points @ WORLD_TO_GL_ROTATION_MAT * UNIT_TO_MM


def camera_to_gl_view(cam):
    """Get the GL view matrix (world-to-camera) from a deltacamera.Camera.

    Returns:
        4x4 view matrix (column-major via transpose)
    """
    R = cam.R
    t = cam.t
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = -R @ t
    return view.T.copy()


def camera_to_gl_mvp(cam, imshape, near=100.0, far=100000.0):
    """Convert deltacamera.Camera to OpenGL MVP matrix.

    Args:
        cam: deltacamera.Camera object
        imshape: (height, width) of the image
        near: near clipping plane in mm
        far: far clipping plane in mm

    Returns:
        4x4 MVP matrix ready for GL (column-major via transpose)
    """
    height, width = imshape[:2]

    # Extract intrinsics
    K = cam.intrinsic_matrix
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Build GL projection matrix from intrinsics
    # Maps camera-space coords to clip coords
    # Camera space: X right, Y down, Z forward (OpenCV convention)
    # We flip Y in the projection to go to GL's Y-up
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2 * fx / width
    proj[0, 2] = 1 - 2 * cx / width
    proj[1, 1] = -2 * fy / height  # Negative to flip Y
    proj[1, 2] = 2 * cy / height - 1
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = 1  # w = z for perspective divide

    # Build view matrix (world to camera)
    # cam.R rows are camera axes in world coords
    # cam.t is camera position in world coords
    R = cam.R  # In deltacamera, R transforms world->camera (rows are camera axes)
    t = cam.t  # Camera position in world (mm)

    # View matrix: transforms world coords to camera coords
    # p_cam = R @ (p_world - t)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[:3, 3] = -R @ t

    mvp = proj @ view

    # Transpose for GLSL column-major order
    return mvp.T.copy()


def _rotation_mat(up):
    """Create rotation matrix from an up vector."""
    up = _unit_vector(up)
    rightlike = np.array([1, 0, 0])
    if np.allclose(up, rightlike):
        rightlike = np.array([0, 1, 0])

    forward = _unit_vector(np.cross(up, rightlike))
    right = np.cross(forward, up)
    return np.row_stack([right, forward, up]).astype(np.float32)


def _unit_vector(v):
    return v / np.linalg.norm(v)
