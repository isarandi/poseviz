import numpy as np
import transforms3d
from mayavi import mlab

import poseviz.cameralib

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
    right = np.array([1, 0, 0])
    if np.allclose(up, right):
        right = np.array([0, 1, 0])

    forward = np.cross(up, right)
    return np.row_stack([right, forward, up])


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


def set_view_to_camera(camera, pivot=None, image_size=None, allow_roll=True):
    if pivot is None:
        pivot = camera.t + camera.R[2] * 500

    fig = mlab.gcf()
    mayavi_cam = fig.scene.camera
    if image_size is not None:
        half_height = 0.5 * image_size[1]
        offset = np.abs(camera.intrinsic_matrix[1, 2] - half_height)
        mayavi_cam.view_angle = np.rad2deg(
            2 * np.arctan(1.2 * (half_height + offset) / camera.intrinsic_matrix[1, 1]))

    mayavi_cam.focal_point = world_to_mayavi(pivot)
    mayavi_cam.position = world_to_mayavi(camera.t)
    if allow_roll:
        mayavi_cam.view_up = -WORLD_TO_MAYAVI_ROTATION_MAT @ camera.R[1]
    mayavi_cam.compute_view_plane_normal()
    fig.scene.renderer.reset_camera_clipping_range()


def get_current_view_as_camera():
    azimuth, elevation, distance, focalpoint = mlab.view()

    azi = -azimuth - 90
    elev = elevation - 90
    total_rotation_mat = transforms3d.euler.euler2mat(azi, elev, 0, 'szxy')
    R = WORLD_TO_MAYAVI_ROTATION_MAT.T @ total_rotation_mat @ WORLD_TO_MAYAVI_ROTATION_MAT
    distance = distance * UNIT_TO_MM
    pivot = mayavi_to_world(focalpoint)
    t = pivot - R[2] * distance
    return poseviz.cameralib.Camera(t, R, np.eye(3), None, WORLD_UP)
