import numpy as np
import tvtk.api
from mayavi import mlab
from scipy.spatial.transform import Rotation
import poseviz.colors
import poseviz.mayavi_util


class CameraViz:
    def __init__(
        self, camera_type, show_image, show_field_of_view=True, show_camera_wireframe=True
    ):
        self.viz_im = None
        self.is_initialized = False
        self.prev_cam = None
        self.camera_type = camera_type
        self.show_image = show_image
        self.prev_imshape = None
        self.show_field_of_view = show_field_of_view
        self.show_camera_wireframe = show_camera_wireframe
        self.mesh = None
        self.mesh2 = None
        self.mesh3 = None
        self.prev_highlight = False

    def initial_update(self, camera, image, highlight=False):
        image_corners, far_corners = self.calculate_camera_vertices(camera, image.shape)

        if self.camera_type != "original":
            triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])

            if self.show_camera_wireframe:
                self.mesh = mlab.triangular_mesh(
                    *image_corners[:5].T,
                    triangles,
                    color=poseviz.colors.cyan if highlight else poseviz.colors.black,
                    tube_radius=0.04 if highlight else 0.02,
                    line_width=10 if highlight else 1,
                    tube_sides=3,
                    representation="wireframe",
                    reset_zoom=False,
                )

            if self.show_field_of_view:
                self.mesh2 = mlab.triangular_mesh(
                    *far_corners.T,
                    triangles,
                    color=poseviz.colors.black,
                    opacity=0.1,
                    representation="surface",
                    reset_zoom=False,
                )
                self.mesh3 = mlab.triangular_mesh(
                    *far_corners.T,
                    triangles,
                    color=poseviz.colors.gray,
                    opacity=0.1,
                    tube_radius=0.01,
                    tube_sides=3,
                    representation="wireframe",
                    reset_zoom=False,
                )

        if self.show_image:
            self.new_imshow(image.shape, image_corners)
            self.set_image_content(image)
            self.set_image_position(image_corners)

        self.prev_cam = camera
        self.prev_imshape = image.shape
        self.prev_highlight = highlight
        self.is_initialized = True

    def update(self, camera, image, highlight=False):
        if not self.is_initialized:
            return self.initial_update(camera, image, highlight)

        if (
            self.prev_cam.allclose(camera)
            and image.shape == self.prev_imshape
            and self.prev_highlight == highlight
        ):
            # Only the image content has changed
            if self.show_image:
                self.set_image_content(image)
            return
        elif self.prev_highlight == highlight:
            image_points, far_corners = self.calculate_camera_vertices(camera, image.shape)
            if self.camera_type != "original":
                if self.show_camera_wireframe:
                    self.mesh.mlab_source.set(
                        x=image_points[:5, 0], y=image_points[:5, 1], z=image_points[:5, 2]
                    )
                if self.show_field_of_view:
                    self.mesh2.mlab_source.set(
                        x=far_corners[:, 0], y=far_corners[:, 1], z=far_corners[:, 2]
                    )
                    self.mesh3.mlab_source.set(
                        x=far_corners[:, 0], y=far_corners[:, 1], z=far_corners[:, 2]
                    )

            if self.show_image:
                if image.shape[:2] != self.prev_imshape[:2] or not np.allclose(
                    camera.intrinsic_matrix, self.prev_cam.intrinsic_matrix
                ):
                    self.viz_im.remove()
                    self.new_imshow(image.shape, image_points)
                    self.prev_imshape = image.shape

                self.set_image_content(image)
                self.set_image_position(image_points)
                self.prev_cam = camera
        else:
            # We change the highlight state by reinitializing it all
            for elem in [self.viz_im, self.mesh, self.mesh2, self.mesh3]:
                if elem is not None:
                    elem.remove()
            self.initial_update(camera, image, highlight)

    def new_imshow(self, imshape, mayavi_image_points):
        mayavi_width = np.linalg.norm(mayavi_image_points[1] - mayavi_image_points[2])
        mayavi_height = np.linalg.norm(mayavi_image_points[2] - mayavi_image_points[3])
        extent = [0, mayavi_height, 0, mayavi_width, 0, 0]
        self.viz_im = mlab.imshow(
            np.ones(imshape[:2]), opacity=0.5, extent=extent, reset_zoom=False, interpolate=True
        )

    def calculate_camera_vertices(self, camera, imshape):
        h, w = imshape[:2]
        image_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        image_center = np.array(imshape[:2][::-1]) / 2
        image_points = np.concatenate([image_corners, [image_center]], axis=0)
        image_points_world = camera.image_to_world(image_points, camera_depth=150)
        points = np.array([camera.t, *image_points_world])
        mayavi_image_points = poseviz.mayavi_util.world_to_mayavi(points)
        mayavi_far_corners = (
            mayavi_image_points[0] + (mayavi_image_points[:5] - mayavi_image_points[0]) * 20
        )
        return mayavi_image_points, mayavi_far_corners

    def set_image_content(self, image):
        reshaped = image.view().reshape([-1, 3], order="F")
        self.colors = tvtk.api.tvtk.UnsignedCharArray()
        self.colors.from_array(reshaped)
        self.viz_im.actor.input.point_data.scalars = self.colors

    def set_image_position(self, mayavi_image_points):
        down = unit_vector(mayavi_image_points[4] - mayavi_image_points[1])
        right = unit_vector(mayavi_image_points[2] - mayavi_image_points[1])
        R = np.column_stack([down, right, np.cross(down, right)])
        z, x, y = Rotation.from_matrix(R).as_euler("ZXY", degrees=True)
        self.viz_im.actor.orientation = [x, y, z]
        self.viz_im.actor.position = mayavi_image_points[5]

    def remove(self):
        if self.is_initialized:
            for elem in [self.viz_im, self.mesh, self.mesh2, self.mesh3]:
                if elem is not None:
                    elem.remove()
            self.is_initialized = False


def unit_vector(v):
    return v / np.linalg.norm(v)
