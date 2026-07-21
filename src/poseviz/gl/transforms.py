import numpy as np


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


def up_basis(world_up):
    """Build an orthonormal world basis from an up vector.

    Returns:
        (right, forward, up): unit vectors such that right x forward = up-ish
        span of the horizontal plane, with a canonical choice that yields
        right=(1,0,0), forward=(0,0,1) for the default Y-down up=(0,-1,0).
    """
    up = _unit_vector(np.asarray(world_up, np.float32))
    rightlike = np.array([1, 0, 0], np.float32)
    if abs(np.dot(up, rightlike)) > 0.99:
        rightlike = np.array([0, 1, 0], np.float32)

    forward = _unit_vector(np.cross(up, rightlike))
    right = np.cross(forward, up)
    return right, forward, up


def _unit_vector(v):
    return v / np.linalg.norm(v)
