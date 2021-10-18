import numpy as np
import transforms3d
import tvtk.api
from mayavi import mlab

import poseviz.boxlib
import poseviz.mayavi_util


class CameraViz:
    def __init__(self, camera_type, show_image, show_field_of_view=True):
        self.viz_im = None
        self.is_initialized = False
        self.prev_cam = None
        self.camera_type = camera_type
        self.show_image = show_image
        self.prev_imshape = None
        self.show_field_of_view = show_field_of_view
        self.mesh = None
        self.mesh2 = None
        self.mesh3 = None

    def initial_update(self, camera, image):
        image_corners, far_corners = self.calculate_camera_vertices(camera, image.shape)

        if self.camera_type != 'original':
            triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
            self.mesh = mlab.triangular_mesh(
                *image_corners.T, triangles, color=(0., 0., 0.), tube_radius=0.02,
                line_width=1, tube_sides=3, representation='wireframe', reset_zoom=False)

            if self.show_field_of_view:
                self.mesh2 = mlab.triangular_mesh(
                    *far_corners.T, triangles, color=(0., 0., 0.), opacity=0.1,
                    representation='surface', reset_zoom=False)
                self.mesh3 = mlab.triangular_mesh(
                    *far_corners.T, triangles, color=(0.5, 0.5, 0.5), opacity=0.1,
                    tube_radius=0.01, tube_sides=3, representation='wireframe', reset_zoom=False)

        if self.show_image:
            self.new_imshow(image.shape, image_corners)
            self.set_image_content(image)
            self.set_image_position(image_corners)

        self.prev_cam = camera
        self.prev_imshape = image.shape
        self.is_initialized = True

    def update(self, camera, image):
        if not self.is_initialized:
            return self.initial_update(camera, image)

        if self.prev_cam.allclose(camera) and image.shape == self.prev_imshape:
            if self.show_image:
                self.set_image_content(image)
            return

        image_corners, far_corners = self.calculate_camera_vertices(camera, image.shape)
        if self.camera_type != 'original':
            self.mesh.mlab_source.set(
                x=image_corners[:, 0], y=image_corners[:, 1], z=image_corners[:, 2])
            if self.show_field_of_view:
                self.mesh2.mlab_source.set(
                    x=far_corners[:, 0], y=far_corners[:, 1], z=far_corners[:, 2])
                self.mesh3.mlab_source.set(
                    x=far_corners[:, 0], y=far_corners[:, 1], z=far_corners[:, 2])

        if self.show_image:
            if image.shape[:2] != self.prev_imshape[:2]:
                self.viz_im.remove()
                self.new_imshow(image.shape, image_corners)
                self.prev_imshape = image.shape

            self.set_image_content(image)
            self.set_image_position(image_corners)
        self.prev_cam = camera

    def new_imshow(self, imshape, mayavi_image_corners):
        mayavi_width = np.linalg.norm(mayavi_image_corners[1] - mayavi_image_corners[2])
        mayavi_height = np.linalg.norm(mayavi_image_corners[2] - mayavi_image_corners[3])
        extent = [0, mayavi_height, 0, mayavi_width, 0, 0]
        self.viz_im = mlab.imshow(
            np.ones(imshape[:2]), opacity=0.6, extent=extent, reset_zoom=False, interpolate=True)

    def calculate_camera_vertices(self, camera, imshape):
        image_corners = poseviz.boxlib.corners(
            poseviz.boxlib.full_box(imshape=np.array(imshape)))
        image_corners_world = camera.image_to_world(image_corners, camera_depth=500)
        points = np.array([camera.t, *image_corners_world])
        mayavi_image_corners = poseviz.mayavi_util.world_to_mayavi(points)
        mayavi_far_corners = (
                mayavi_image_corners[0] + (mayavi_image_corners - mayavi_image_corners[0]) * 20)
        return mayavi_image_corners, mayavi_far_corners

    def set_image_content(self, image):
        reshaped = image.view().reshape([-1, 3], order='F')
        self.colors = tvtk.api.tvtk.UnsignedCharArray()
        self.colors.from_array(reshaped)
        self.viz_im.actor.input.point_data.scalars = self.colors

    def set_image_position(self, mayavi_image_corners):
        down = unit_vector(mayavi_image_corners[4] - mayavi_image_corners[1])
        right = unit_vector(mayavi_image_corners[2] - mayavi_image_corners[1])
        R = np.column_stack([down, right, np.cross(down, right)])
        z, x, y = np.rad2deg(transforms3d.euler.mat2euler(R, 'rzxy'))
        self.viz_im.actor.orientation = [x, y, z]
        self.viz_im.actor.position = np.mean(mayavi_image_corners[1:], axis=0)


def unit_vector(v):
    return v / np.linalg.norm(v)
