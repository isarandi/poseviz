import copy

import cv2
import numpy as np


def support_single(f):
    """Makes a function that transforms multiple points accept also a single point"""

    def wrapped(self, points, *args, **kwargs):
        ndim = np.array(points).ndim
        if ndim == 1:
            return f(self, np.array([points]), *args, **kwargs)[0]
        else:
            return f(self, points, *args, **kwargs)

    return wrapped


class Camera:
    def __init__(
            self, optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
            distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None):
        """Initializes camera.

        The camera coordinate system has the following axes:
          x points to the right
          y points down
          z points forwards

        The world z direction is assumed to point up by default, but `world_up` can also be
         specified differently.

        Args:
            optical_center: position of the camera in world coordinates (eye point)
            rot_world_to_cam: 3x3 rotation matrix for transforming column vectors
                from being expressed in world reference frame to being expressed in camera
                reference frame as follows:
                column_point_cam = rot_matrix_world_to_cam @ (column_point_world - optical_center)
            intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
                coordinates in image (pixel) space. Its last row must be (0,0,1).
            distortion_coeffs: parameters describing radial and tangential lens distortions,
                following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
                if the camera has no distortion.
            world_up: a world vector that is designated as "pointing up", for use when
                the camera wants to roll itself upright.
        """

        if optical_center is not None and extrinsic_matrix is not None:
            raise Exception('At most one of `optical_center` and `extrinsic_matrix` needs to be '
                            'provided!')
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise Exception('At most one of `rot_world_to_cam` and `extrinsic_matrix` needs to be '
                            'provided!')

        if (optical_center is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3)

        if extrinsic_matrix is not None:
            self.R = np.asarray(extrinsic_matrix[:3, :3], np.float32)
            self.t = (-self.R.T @ extrinsic_matrix[:3, 3]).astype(np.float32)
        else:
            self.R = np.asarray(rot_world_to_cam, np.float32)
            self.t = np.asarray(optical_center, np.float32)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, np.float32)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, np.float32)

        self.world_up = np.asarray(world_up)

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise Exception(f'Bottom row of camera\'s intrinsic matrix must be (0,0,1), '
                            f'got {self.intrinsic_matrix[2, :]}.')

    @staticmethod
    def create2D(imshape=(0, 0)):
        intrinsics = np.eye(3)
        intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]
        return Camera([0, 0, 0], np.eye(3), intrinsics, None)

    def allclose(self, other_camera):
        return (np.allclose(self.intrinsic_matrix, other_camera.intrinsic_matrix) and
                np.allclose(self.R, other_camera.R) and np.allclose(self.t, other_camera.t) and
                allclose_or_nones(self.distortion_coeffs, other_camera.distortion_coeffs))

    @support_single
    def camera_to_image(self, points):
        """Transforms points from 3D camera coordinate space to image space.
        The steps involved are:
            1. Projection
            2. Distortion (radial and tangential)
            3. Applying focal length and principal point (intrinsic matrix)

        Equivalently:

        projected = points[:, :2] / points[:, 2:]

        if self.distortion_coeffs is not None:
            r2 = np.sum(projected[:, :2] ** 2, axis=1, keepdims=True)

            k = self.distortion_coeffs[[0, 1, 4]]
            radial = 1 + np.hstack([r2, r2 ** 2, r2 ** 3]) @ k

            p_flipped = self.distortion_coeffs[[3, 2]]
            tagential = projected @ (p_flipped * 2)
            distorted = projected * np.expand_dims(radial + tagential, -1) + p_flipped * r2
        else:
            distorted = projected

        return distorted @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]
        """
        # points = np.asarray(points, np.float32)
        # zeros = np.zeros(3, np.float32)
        # return cv2.projectPoints(
        #     np.expand_dims(points, 0), zeros, zeros, self.intrinsic_matrix,
        #     self.distortion_coeffs)[0][:, 0, :]
        #
        # points = np.asarray(points, np.float32)

        if self.distortion_coeffs is not None:
            result = project_points(points, self.distortion_coeffs, self.intrinsic_matrix)
            return result
        else:
            projected = points[:, :2] / points[:, 2:]
            return projected @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]

        # zeros = np.zeros(3, np.float32)
        # return cv2.projectPoints(
        #     np.expand_dims(points, 0), zeros, zeros, self.intrinsic_matrix,
        #     self.distortion_coeffs)[0][:, 0, :]

    @support_single
    def world_to_camera(self, points):
        points = np.asarray(points, np.float32)
        return (points - self.t) @ self.R.T

    @support_single
    def camera_to_world(self, points):
        points = np.asarray(points, np.float32)
        # Here we need to use R inverse but we are also
        # operating on row-vector points, so we need to
        # transpose as well, which both cancel out, leaving just R.
        return points @ self.R + self.t

    @support_single
    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    @support_single
    def image_to_camera(self, points, depth=1):
        points = np.expand_dims(np.asarray(points, np.float32), 0)
        new_image_points = cv2.undistortPoints(
            points, self.intrinsic_matrix, self.distortion_coeffs, None, None, None)
        return cv2.convertPointsToHomogeneous(new_image_points)[:, 0, :] * depth

    @support_single
    def image_to_world(self, points, camera_depth=1):
        return self.camera_to_world(self.image_to_camera(points, camera_depth))

    def is_visible(self, world_points, imsize):
        imsize = np.asarray(imsize)
        cam_points = self.world_to_camera(world_points)
        im_points = self.camera_to_image(cam_points)

        is_within_frame = np.all(np.logical_and(0 <= im_points, im_points < imsize), axis=1)
        is_in_front_of_camera = cam_points[..., 2] > 0
        return np.logical_and(is_within_frame, is_in_front_of_camera)

    def zoom(self, factor):
        """Zooms the camera (factor > 1 makes objects look larger),
        while keeping the principal point fixed (scaling anchor is the principal point)."""
        self.intrinsic_matrix[:2, :2] *= np.expand_dims(factor, -1)

    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(factor, -1)

    def undistort(self):
        self.distortion_coeffs = None

    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]])
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    def horizontal_flip(self):
        self.R[0] *= -1

    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = [imshape[1] / 2, imshape[0] / 2]

    def shift_to_center(self, desired_center_image_point, imshape):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        current_coords_of_the_point = desired_center_image_point
        target_coords_of_the_point = np.float32([imshape[1], imshape[0]]) / 2
        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    def shift_to_desired(self, current_coords_of_the_point, target_coords_of_the_point):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    def turn_towards(self, target_image_point=None, target_world_point=None):
        """Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll)."""

        assert (target_image_point is None) != (target_world_point is None)
        if target_image_point is not None:
            target_world_point = self.image_to_world(target_image_point)

        def unit_vec(v):
            return v / np.linalg.norm(v)

        new_z = unit_vec(target_world_point - self.t)
        new_x = unit_vec(np.cross(new_z, self.world_up))
        new_y = np.cross(new_z, new_x)

        # row_stack because we need the inverse transform (we make a matrix that transforms
        # points from one coord system to another), which is the same as the transpose
        # for rotation matrices.
        self.R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)

    def reset_roll(self):
        def unit_vec(v):
            return v / np.linalg.norm(v)

        self.R[:, 0] = unit_vec(np.cross(self.R[:, 2], self.world_up))
        self.R[:, 1] = np.cross(self.R[:, 0], self.R[:, 2])

    def orbit_around(self, world_point, angle_radians, axis='vertical'):
        """Rotates the camera around a vertical axis passing through `world point` by
        `angle_radians`."""

        if axis == 'vertical':
            axis = self.world_up
        else:
            lookdir = self.R[2]
            axis = np.cross(lookdir, self.world_up)
            axis = axis / np.linalg.norm(axis)

        rot_matrix = cv2.Rodrigues(axis * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = (rot_matrix @ (self.t - world_point)) + world_point

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = self.R @ rot_matrix.T

    def crop_from(self, point):
        self.intrinsic_matrix[:2, 2] -= point

    def get_projection_matrix(self):
        extrinsic_projection = np.append(self.R, -self.R @ np.expand_dims(self.t, 1), axis=1)
        return self.intrinsic_matrix @ extrinsic_projection

    def get_extrinsic_matrix(self):
        return build_extrinsic_matrix(self.R, self.t)

    @staticmethod
    def from_fov(fov_degrees, imshape):
        f = np.max(imshape[:2]) / (np.tan(np.deg2rad(fov_degrees) / 2) * 2)
        intrinsics = np.array(
            [[f, 0, imshape[1] / 2],
             [0, f, imshape[0] / 2],
             [0, 0, 1]], np.float32)
        return Camera(intrinsic_matrix=intrinsics)

    def copy(self):
        return copy.deepcopy(self)


def build_extrinsic_matrix(rot_world_to_cam, optical_center_world):
    R = rot_world_to_cam
    t = optical_center_world
    return np.block([[R, -R @ np.expand_dims(t, -1)], [0, 0, 0, 1]])


def allclose_or_nones(a, b):
    if a is None and b is None:
        return True

    if a is None:
        return np.min(b) == np.max(b) == 0

    if b is None:
        return np.min(b) == np.max(b) == 0

    return np.allclose(a, b)


def project_points(points, dist_coeff, intrinsic_matrix):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    points = points.astype(np.float32)
    proj = points[:, :2] / points[:, 2:]
    r2 = np.sum(proj * proj, axis=1)
    distorter = (
            ((dist_coeff[4] * r2 + dist_coeff[1]) * r2 + dist_coeff[0]) * r2 +
            np.float32(1.0) + np.sum(proj * (np.float32(2.0) * dist_coeff[3:1:-1]), axis=1))
    proj[:] = (
            proj * np.expand_dims(distorter, 1) + np.expand_dims(r2, 1) * dist_coeff[3:1:-1])
    return (proj @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)
