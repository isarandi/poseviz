import logging

import poseviz.components.camera_viz
import poseviz.colors as colors
import poseviz.components.skeletons_viz


class MainViz:
    def __init__(
            self, joint_info_pred, joint_info_pred2, joint_info_gt, camera_type, show_image,
            high_quality, show_field_of_view=True):
        if joint_info_pred is not None:
            self.skeletons_pred = poseviz.components.skeletons_viz.SkeletonsViz(
                joint_info_pred, colors.blue, colors.cyan, colors.yellow, colors.green, 0.06,
                high_quality)
        else:
            self.skeletons_pred = None

        if joint_info_gt is not None:
            self.skeletons_gt = poseviz.components.skeletons_viz.SkeletonsViz(
                joint_info_gt, colors.red, colors.red, colors.red, colors.red, 0.03, high_quality)
        else:
            self.skeletons_gt = None

        if joint_info_pred2 is not None:
            self.skeletons_pred2 = poseviz.components.skeletons_viz.SkeletonsViz(
                joint_info_pred2, colors.blue, colors.cyan, colors.yellow, colors.orange, 0.06,
                high_quality)
        else:
            self.skeletons_pred2 = None

        self.camera_viz = poseviz.components.camera_viz.CameraViz(
            camera_type, show_image, show_field_of_view)

    def update(self, camera_display, image, pred_poses=None, pred_poses2=None, gt_poses=None):
        if pred_poses is not None:
            self.skeletons_pred.update(pred_poses)
        if pred_poses2 is not None:
            self.skeletons_pred2.update(pred_poses2)
        if gt_poses is not None:
            self.skeletons_gt.update(gt_poses)

        self.camera_viz.update(camera_display, image)
