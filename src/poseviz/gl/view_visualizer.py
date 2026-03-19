import logging

import numpy as np
import poseviz.colors as colors

logger = logging.getLogger(__name__)
from poseviz.gl.renderables import (
    SphereRenderable,
    TubeRenderable,
    MeshRenderable,
    ImageQuadRenderable,
    WireframeRenderable,
    UniformColor,
)


class ViewVisualizer:
    """Visualizes one camera's data: image plane, skeleton poses, and meshes."""

    def __init__(
        self,
        ctx,
        joint_info,
        camera_type,
        show_image=True,
        high_quality=True,
        show_field_of_view=True,
        show_camera_wireframe=True,
        body_model_faces=None,
        image_plane_distance=1000,
    ):
        self.ctx = ctx
        self.joint_names, self.joint_edges = joint_info
        self.camera_type = camera_type
        self.show_image = show_image
        self.show_field_of_view = show_field_of_view
        self.show_camera_wireframe = show_camera_wireframe
        self.image_plane_distance = image_plane_distance

        # Classify joints by side for coloring
        if self.joint_names is not None:
            self.left_indices = [
                i for i, n in enumerate(self.joint_names) if n.startswith("l")
            ]
            self.mid_indices = [
                i
                for i, n in enumerate(self.joint_names)
                if not n.startswith(("l", "r"))
            ]
            self.right_indices = [
                i for i, n in enumerate(self.joint_names) if n.startswith("r")
            ]

            # Classify edges by side
            self.left_edges = []
            self.mid_edges = []
            self.right_edges = []
            for i, j in self.joint_edges:
                n1, n2 = self.joint_names[i], self.joint_names[j]
                if n1.startswith("l") and n2.startswith("l"):
                    self.left_edges.append((i, j))
                elif n1.startswith("r") and n2.startswith("r"):
                    self.right_edges.append((i, j))
                else:
                    self.mid_edges.append((i, j))

        # Skeleton renderers (pred, true, alt)
        resolution = 16 if high_quality else 4
        sides = 12 if high_quality else 4
        scale = 30.0  # Joint sphere radius in mm

        if self.joint_names is not None:
            # Predicted poses - blue/cyan/yellow by side
            self.spheres_pred_left = SphereRenderable(ctx, colors.blue, resolution)
            self.spheres_pred_mid = SphereRenderable(ctx, colors.cyan, resolution)
            self.spheres_pred_right = SphereRenderable(ctx, colors.yellow, resolution)
            self.tubes_pred_left = TubeRenderable(ctx, colors.blue, sides)
            self.tubes_pred_mid = TubeRenderable(ctx, colors.green, sides)
            self.tubes_pred_right = TubeRenderable(ctx, colors.yellow, sides)

            # True poses - red
            self.spheres_true = SphereRenderable(ctx, colors.red, resolution)
            self.tubes_true = TubeRenderable(ctx, colors.red, sides)

            # Alt poses - orange
            self.spheres_alt = SphereRenderable(ctx, colors.orange, resolution)
            self.tubes_alt = TubeRenderable(ctx, colors.orange, sides)
        else:
            self.spheres_pred_left = None

        # Mesh renderers (dynamic lists - one per body)
        self.body_model_faces = body_model_faces
        self.meshes_pred = []
        self.meshes_true = []
        self.meshes_alt = []

        # Camera visualization
        if show_image:
            self.image_renderer = ImageQuadRenderable(ctx, opacity=0.5)
        else:
            self.image_renderer = None

        self.highlight = False
        if show_camera_wireframe and camera_type != "original":
            self.camera_wireframe = WireframeRenderable(ctx, colors.black)
            self.camera_wireframe_tubes = TubeRenderable(ctx, colors.cyan, sides=sides)
            self.camera_wireframe_spheres = SphereRenderable(ctx, colors.cyan, resolution)
        else:
            self.camera_wireframe = None
            self.camera_wireframe_tubes = None
            self.camera_wireframe_spheres = None

        self.scale = scale

    def update(
        self,
        camera,
        frame,
        poses=None,
        poses_true=None,
        poses_alt=None,
        vertices=None,
        vertices_true=None,
        vertices_alt=None,
        highlight=False,
    ):
        """Update visualizer with new data."""
        import time
        _t0 = time.perf_counter()

        # Update image
        if self.image_renderer and frame is not None:
            self.image_renderer.update(camera, frame, self.image_plane_distance)

        _t1 = time.perf_counter()

        # Update camera wireframe
        if self.camera_wireframe and frame is not None:
            self.highlight = highlight
            if highlight:
                vertices, starts, ends = _pyramid_geometry(
                    camera, frame.shape, self.image_plane_distance
                )
                radius = 15.0
                self.camera_wireframe_spheres.update(vertices, scale=radius)
                self.camera_wireframe_tubes.update(starts, ends, radius=radius)
            else:
                self.camera_wireframe.update_pyramid(
                    camera, frame.shape, self.image_plane_distance
                )

        _t2 = time.perf_counter()

        # Update skeletons
        if poses is not None and self.spheres_pred_left is not None:
            self._update_skeleton(
                poses,
                self.spheres_pred_left,
                self.spheres_pred_mid,
                self.spheres_pred_right,
                self.tubes_pred_left,
                self.tubes_pred_mid,
                self.tubes_pred_right,
            )

        if poses_true is not None and self.spheres_pred_left is not None:
            self._update_skeleton_single_color(
                poses_true, self.spheres_true, self.tubes_true
            )

        if poses_alt is not None and self.spheres_pred_left is not None:
            self._update_skeleton_single_color(
                poses_alt, self.spheres_alt, self.tubes_alt
            )

        _t3 = time.perf_counter()

        # Update meshes
        if vertices is not None and self.body_model_faces is not None:
            self._update_meshes(vertices, self.meshes_pred, colors.blue)

        if vertices_true is not None and self.body_model_faces is not None:
            self._update_meshes(vertices_true, self.meshes_true, colors.red)

        if vertices_alt is not None and self.body_model_faces is not None:
            self._update_meshes(vertices_alt, self.meshes_alt, colors.orange)

        _t4 = time.perf_counter()

        logger.debug(
            f"    [viz.update] image={(_t1-_t0)*1000:.1f}ms "
            f"wireframe={(_t2-_t1)*1000:.1f}ms "
            f"skeleton={(_t3-_t2)*1000:.1f}ms "
            f"mesh={(_t4-_t3)*1000:.1f}ms"
        )

    def _update_skeleton(
        self,
        poses,
        spheres_left,
        spheres_mid,
        spheres_right,
        tubes_left,
        tubes_mid,
        tubes_right,
    ):
        """Update skeleton renderers with per-side coloring."""
        if not poses:
            spheres_left.update(np.zeros((0, 3), np.float32))
            spheres_mid.update(np.zeros((0, 3), np.float32))
            spheres_right.update(np.zeros((0, 3), np.float32))
            tubes_left.update(np.zeros((0, 3)), np.zeros((0, 3)))
            tubes_mid.update(np.zeros((0, 3)), np.zeros((0, 3)))
            tubes_right.update(np.zeros((0, 3)), np.zeros((0, 3)))
            return

        all_joints = np.concatenate(
            poses, axis=0
        )  # (total_joints, 3) in world coords (mm)
        n_joints = poses[0].shape[0]
        n_poses = len(poses)

        # Reshape for indexing
        joints_reshaped = all_joints.reshape(n_poses, n_joints, 3)

        # Gather joints by side
        left_joints = joints_reshaped[:, self.left_indices].reshape(-1, 3)
        mid_joints = joints_reshaped[:, self.mid_indices].reshape(-1, 3)
        right_joints = joints_reshaped[:, self.right_indices].reshape(-1, 3)

        spheres_left.update(left_joints, self.scale)
        spheres_mid.update(mid_joints, self.scale)
        spheres_right.update(right_joints, self.scale)

        # Gather tube endpoints by side
        def gather_edges(edges):
            starts = []
            ends = []
            for pose in joints_reshaped:
                for i, j in edges:
                    starts.append(pose[i])
                    ends.append(pose[j])
            if starts:
                return np.array(starts, np.float32), np.array(ends, np.float32)
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        tubes_left.update(*gather_edges(self.left_edges), self.scale / 5)
        tubes_mid.update(*gather_edges(self.mid_edges), self.scale / 5)
        tubes_right.update(*gather_edges(self.right_edges), self.scale / 5)

    def _update_skeleton_single_color(self, poses, spheres, tubes):
        """Update skeleton renderers with single color."""
        if not poses:
            spheres.update(np.zeros((0, 3), np.float32))
            tubes.update(np.zeros((0, 3)), np.zeros((0, 3)))
            return

        all_joints = np.concatenate(poses, axis=0)  # World coords (mm)

        spheres.update(all_joints, self.scale)

        # All edges
        n_joints = poses[0].shape[0]
        joints_reshaped = all_joints.reshape(len(poses), n_joints, 3)

        starts = []
        ends = []
        for pose in joints_reshaped:
            for i, j in self.joint_edges:
                starts.append(pose[i])
                ends.append(pose[j])

        if starts:
            tubes.update(
                np.array(starts, np.float32), np.array(ends, np.float32), self.scale / 5
            )
        else:
            tubes.update(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32))

    def _update_meshes(self, vertices_list, mesh_renderers, color):
        """Update mesh renderers with vertices (one renderer per body)."""
        n_bodies = len(vertices_list)

        # Create new renderers if needed
        while len(mesh_renderers) < n_bodies:
            mesh_renderers.append(
                MeshRenderable(self.ctx, self.body_model_faces, UniformColor(color))
            )

        # Destroy excess renderers
        while len(mesh_renderers) > n_bodies:
            mesh_renderers.pop().destroy()

        # Update each renderer with its body's vertices
        for renderer, verts in zip(mesh_renderers, vertices_list):
            if verts.shape[-1] == 4:
                verts = verts[:, :3]  # Strip uncertainty
            renderer.update(verts)

    def render(self, view_proj):
        """Render all components."""
        # Render skeletons first (before transparent image)
        if self.spheres_pred_left is not None:
            self.spheres_pred_left.render(view_proj)
            self.spheres_pred_mid.render(view_proj)
            self.spheres_pred_right.render(view_proj)
            self.tubes_pred_left.render(view_proj)
            self.tubes_pred_mid.render(view_proj)
            self.tubes_pred_right.render(view_proj)
            self.spheres_true.render(view_proj)
            self.tubes_true.render(view_proj)
            self.spheres_alt.render(view_proj)
            self.tubes_alt.render(view_proj)

        # Render meshes
        for mesh in self.meshes_pred:
            mesh.render(view_proj)
        for mesh in self.meshes_true:
            mesh.render(view_proj)
        for mesh in self.meshes_alt:
            mesh.render(view_proj)

        # Render transparent image last (on top of skeleton)
        if self.image_renderer:
            self.image_renderer.render(view_proj)

        # Render camera wireframe (tubes+spheres when highlighted, lines otherwise)
        if self.camera_wireframe:
            if self.highlight:
                self.camera_wireframe_tubes.render(view_proj)
                self.camera_wireframe_spheres.render(view_proj)
            else:
                self.camera_wireframe.render(view_proj)

    def destroy(self):
        """Release all GPU resources."""
        if self.image_renderer:
            self.image_renderer.destroy()
        if self.camera_wireframe:
            self.camera_wireframe.destroy()
            self.camera_wireframe_tubes.destroy()
            self.camera_wireframe_spheres.destroy()
        if self.spheres_pred_left:
            self.spheres_pred_left.destroy()
            self.spheres_pred_mid.destroy()
            self.spheres_pred_right.destroy()
            self.tubes_pred_left.destroy()
            self.tubes_pred_mid.destroy()
            self.tubes_pred_right.destroy()
            self.spheres_true.destroy()
            self.tubes_true.destroy()
            self.spheres_alt.destroy()
            self.tubes_alt.destroy()
        for mesh in self.meshes_pred:
            mesh.destroy()
        for mesh in self.meshes_true:
            mesh.destroy()
        for mesh in self.meshes_alt:
            mesh.destroy()


def _pyramid_geometry(camera, image_shape, image_plane_distance):
    """Compute camera pyramid vertices and line segments for tube/sphere rendering.

    Returns:
        vertices: (5, 3) apex + 4 corners
        starts: (8, 3) line segment start points
        ends: (8, 3) line segment end points
    """
    h, w = image_shape[:2]
    corners_2d = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    corners = camera.image_to_world(corners_2d, camera_depth=image_plane_distance)
    apex = camera.t.reshape(1, 3)
    vertices = np.concatenate([apex, corners], axis=0).astype(np.float32)

    starts = []
    ends = []
    # Apex to each corner
    for corner in corners:
        starts.append(apex[0])
        ends.append(corner)
    # Rectangle edges
    for i in range(4):
        starts.append(corners[i])
        ends.append(corners[(i + 1) % 4])

    return vertices, np.array(starts, np.float32), np.array(ends, np.float32)
