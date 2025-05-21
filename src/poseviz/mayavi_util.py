import numpy as np
import transforms3d
from mayavi import mlab

import cameravision

MM_TO_UNIT = 1 / 1000
UNIT_TO_MM = 1000
WORLD_UP = np.array([0, 0, 1])
WORLD_TO_MAYAVI_ROTATION_MAT = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]], dtype=np.float32)

CAM_TO_MAYCAM_ROTATION_MAT = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1]], dtype=np.float32)


def rotation_mat(up):
    up = unit_vector(up)
    rightlike = np.array([1, 0, 0])
    if np.allclose(up, rightlike):
        rightlike = np.array([0, 1, 0])

    forward = unit_vector(np.cross(up, rightlike))
    right = np.cross(forward, up)
    return np.row_stack([right, forward, up])


def unit_vector(vectors, axis=-1):
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / norm


def world_to_mayavi(points):
    points = np.asarray(points)
    if points.ndim == 1:
        return WORLD_TO_MAYAVI_ROTATION_MAT @ points * MM_TO_UNIT
    else:
        return points @ WORLD_TO_MAYAVI_ROTATION_MAT.T * MM_TO_UNIT


def mayavi_to_world(points):
    points = np.asarray(points)
    if points.ndim == 1:
        return WORLD_TO_MAYAVI_ROTATION_MAT.T @ points * UNIT_TO_MM
    else:
        return points @ WORLD_TO_MAYAVI_ROTATION_MAT * UNIT_TO_MM


def set_world_up(world_up):
    global WORLD_UP
    WORLD_UP = np.asarray(world_up)
    global WORLD_TO_MAYAVI_ROTATION_MAT
    WORLD_TO_MAYAVI_ROTATION_MAT = rotation_mat(WORLD_UP)


def set_view_to_camera(
        camera, pivot=None, image_size=None, view_angle=None, allow_roll=True,
        camera_view_padding=0.2):
    if pivot is None:
        pivot = camera.t + camera.R[2] * 2500

    fig = mlab.gcf()
    mayavi_cam = fig.scene.camera
    if image_size is not None:
        half_height = 0.5 * image_size[1]
        offset = np.abs(camera.intrinsic_matrix[1, 2] - half_height)
        tan = (1 + camera_view_padding) * (half_height + offset) / camera.intrinsic_matrix[1, 1]
        mayavi_cam.view_angle = np.rad2deg(2 * np.arctan(tan))
    elif view_angle is not None:
        mayavi_cam.view_angle = view_angle

    mayavi_cam.focal_point = world_to_mayavi(pivot)
    mayavi_cam.position = world_to_mayavi(camera.t)
    if allow_roll:
        mayavi_cam.view_up = -WORLD_TO_MAYAVI_ROTATION_MAT @ camera.R[1]
    mayavi_cam.compute_view_plane_normal()
    fig.scene.renderer.reset_camera_clipping_range()


def get_current_view_as_camera():
    azimuth, elevation, distance, focalpoint = mlab.view()
    azi = -azimuth - 90
    elev = -elevation + 90
    total_rotation_mat = transforms3d.euler.euler2mat(np.deg2rad(azi), np.deg2rad(elev), 0, 'szxy')
    R = WORLD_TO_MAYAVI_ROTATION_MAT.T @ total_rotation_mat @ WORLD_TO_MAYAVI_ROTATION_MAT

    fig = mlab.gcf()
    t = mayavi_to_world(fig.scene.camera.position)

    width, height = fig.scene.get_size()
    f = height / (np.tan(np.deg2rad(fig.scene.camera.view_angle) / 2) * 2)
    intrinsics = np.array(
        [[f, 0, (width - 1) / 2],
         [0, f, (height - 1) / 2],
         [0, 0, 1]], np.float32)
    return cameravision.Camera(
        intrinsic_matrix=intrinsics, rot_world_to_cam=R, optical_center=t, world_up=WORLD_UP)
