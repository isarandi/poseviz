import logging

import poseviz.colors as colors
import poseviz.components.camera_viz
import poseviz.components.skeletons_viz
import poseviz.components.smpl_viz


class MainViz:
    def __init__(
        self,
        joint_info_pred,
        joint_info_true,
        joint_info_alt,
        camera_type,
        show_image,
        high_quality,
        show_field_of_view=True,
        show_camera_wireframe=True,
        body_model_faces=None,
    ):

        if joint_info_pred[0] is not None:
            self.skeletons_pred = poseviz.components.skeletons_viz.SkeletonsViz(
                joint_info_pred,
                colors.blue,
                colors.cyan,
                colors.yellow,
                colors.green,
                0.06,
                high_quality,
                opacity=0.95,
            )
        else:
            self.skeletons_pred = None

        if joint_info_true[0] is not None:
            self.skeletons_true = poseviz.components.skeletons_viz.SkeletonsViz(
                joint_info_true,
                colors.red,
                colors.red,
                colors.red,
                colors.red,
                0.06,
                high_quality,
                opacity=0.95,
            )
        else:
            self.skeletons_true = None

        if joint_info_alt[0] is not None:
            self.skeletons_alt = poseviz.components.skeletons_viz.SkeletonsViz(
                joint_info_alt,
                colors.orange,
                colors.orange,
                colors.orange,
                colors.orange,
                0.06,
                high_quality,
                opacity=0.95,
            )
        else:
            self.skeletons_alt = None

        if body_model_faces is not None:
            self.meshes_pred = poseviz.components.smpl_viz.SMPLViz(
                color=colors.blue, faces=body_model_faces, add_wireframe=True, colormap="Blues_r"
            )
            self.meshes_gt = poseviz.components.smpl_viz.SMPLViz(
                color=colors.red, faces=body_model_faces, add_wireframe=True
            )
            self.meshes_pred2 = poseviz.components.smpl_viz.SMPLViz(
                color=colors.orange,
                faces=body_model_faces,
                add_wireframe=True,
                colormap="Oranges_r",
            )
        else:
            self.meshes_pred = None
            self.meshes_gt = None
            self.meshes_pred2 = None

        self.camera_viz = poseviz.components.camera_viz.CameraViz(
            camera_type, show_image, show_field_of_view, show_camera_wireframe
        )

    def update(
        self,
        camera_display,
        image,
        poses_pred=None,
        poses_true=None,
        poses_alt=None,
        vertices_pred=None,
        vertices_true=None,
        vertices_alt=None,
        highlight=False,
    ):

        if poses_pred is not None and self.skeletons_pred is not None:
            self.skeletons_pred.update(poses_pred)
        if poses_true is not None and self.skeletons_true is not None:
            self.skeletons_true.update(poses_true)
        if poses_alt is not None and self.skeletons_alt is not None:
            self.skeletons_alt.update(poses_alt)

        if vertices_pred is not None and self.meshes_pred is not None:
            self.meshes_pred.update(vertices_pred)
        if vertices_true is not None and self.meshes_gt is not None:
            self.meshes_gt.update(vertices_true)
        if vertices_alt is not None and self.meshes_pred2 is not None:
            self.meshes_pred2.update(vertices_alt)

        self.camera_viz.update(camera_display, image, highlight=highlight)

    def remove(self):
        if self.is_initialized:
            for elem in [
                self.skeletons_pred,
                self.skeletons_true,
                self.skeletons_alt,
                self.meshes_pred,
                self.meshes_gt,
                self.meshes_pred2,
            ]:
                if elem is not None:
                    elem.remove()

            self.camera_viz.remove()

            self.is_initialized = False
