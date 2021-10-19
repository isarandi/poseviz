import poseviz.colors
import collections
import itertools
import multiprocessing as mp
import poseviz.video_writing
import queue
from typing import List

import numpy as np

ViewInfo = collections.namedtuple(
    'ViewInfo', ['frame', 'boxes', 'poses', 'camera', 'poses_true', 'poses_alt'],
    defaults=(None, (), (), None, (), ()))


class PoseViz:
    def __init__(
            self, joint_names, joint_edges, camera_type='free', n_views=1, world_up=(0, -1, 0),
            ground_plane_height=-1000, downscale=1, viz_fps=100, queue_size=64, write_video=False,
            multicolor_detections=False, snap_to_cam_on_scene_change=True, high_quality=True,
            draw_2d_pose=False, show_field_of_view=True):

        self.q_posedata = mp.JoinableQueue(queue_size)

        if write_video:
            self.q_out_video_frames = mp.JoinableQueue(queue_size)
            self.video_writer_process = mp.Process(
                target=poseviz.video_writing.main_video_writer,
                args=(self.q_out_video_frames,))
            self.video_writer_process.start()
        else:
            self.q_out_video_frames = None

        self.downscale = downscale
        self.snap_to_cam_on_scene_change = snap_to_cam_on_scene_change

        self.visualizer_process = mp.Process(
            target=_main_visualize, args=(
                self.q_posedata, self.q_out_video_frames, joint_names, joint_edges, camera_type,
                n_views, world_up, ground_plane_height, viz_fps, multicolor_detections,
                snap_to_cam_on_scene_change, high_quality, draw_2d_pose, show_field_of_view))
        self.visualizer_process.start()

    def update(self, frame, boxes, poses, camera, poses_true=(), poses_alt=(), block=True):
        viewinfo = ViewInfo(frame, boxes, poses, camera, poses_true, poses_alt)
        self.update_multiview([viewinfo], block=block)

    def update_multiview(self, view_infos: List[ViewInfo], block=True):
        view_infos = list(map(viewinfo_tf_to_numpy, view_infos))

        d = self.downscale
        if d != 1:
            rescaled_cameras = [v.camera.copy() for v in view_infos]
            for c in rescaled_cameras:
                c.scale_output(1 / d)

            view_infos = [
                ViewInfo(v.frame[d // 2::d, d // 2::d].copy(), v.boxes / d, v.poses, c,
                         v.poses_true, v.poses_alt)
                for v, c in zip(view_infos, rescaled_cameras)]

        try:
            self.q_posedata.put(view_infos, block=block)
        except queue.Full:
            pass

    def new_sequence(self):
        self.q_posedata.put('new_sequence')
        self._join()

    def start_new_video(self, new_video_path, fps):
        self._join()
        self.q_out_video_frames.put((new_video_path, fps))

    def close(self):
        self.q_posedata.put('stop_visualization')  # close mayavi
        self._join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _join(self):
        self.q_posedata.join()
        if self.q_out_video_frames is not None:
            self.q_out_video_frames.join()


class PoseVizMayaviSide:
    def __init__(
            self, q_posedata, q_out_video_frames, joint_names, joint_edges, camera_type, n_views,
            world_up, ground_plane_height, fps, multicolor_detections, snap_to_cam_on_scene_change,
            high_quality, draw_2d_pose, show_field_of_view):

        self.q_posedata = q_posedata
        self.q_out_video_frames = q_out_video_frames

        self.camera_type = camera_type
        self.joint_info = (joint_names, joint_edges)
        self.initialized_camera = False
        self.n_views = n_views
        self.world_up = world_up
        self.multicolor_detections = multicolor_detections
        self.main_cam = 0
        self.pose_displayed_cam_id = None
        self.fps = fps
        self.snap_to_cam_on_scene_change = snap_to_cam_on_scene_change
        self.ground_plane_height = ground_plane_height
        self.step_one_by_one = False
        self.paused = False
        self.current_viewinfos = None
        self.high_quality = high_quality
        self.show_field_of_view = show_field_of_view
        self.draw_2d_pose = draw_2d_pose

    def run_loop(self):
        # Imports are here so Mayavi is loaded in the visualizer process
        from mayavi import mlab
        import poseviz.init
        import poseviz.components.main_viz
        import poseviz.components.ground_plane_viz
        import poseviz.mayavi_util
        fig = poseviz.init.initialize()
        fig.scene.interactor.add_observer('KeyPressEvent', self._on_keypress)
        poseviz.components.ground_plane_viz.draw_checkerboard(
            ground_plane_height=self.ground_plane_height)
        poseviz.mayavi_util.set_world_up(self.world_up)
        self.view_visualizers = [poseviz.components.main_viz.MainViz(
            self.joint_info, self.joint_info, self.joint_info,
            self.camera_type, True, self.high_quality, show_field_of_view=self.show_field_of_view)
            for _ in range(self.n_views)]

        delay = max(10, int(round(1000 / self.fps)))

        @mlab.animate(delay=delay, ui=False)
        def anim():
            while True:
                try:
                    received_info = self.q_posedata.get_nowait() if not self.paused else 'nothing'
                except queue.Empty:
                    received_info = 'nothing'

                if received_info == 'nothing':
                    if self.current_viewinfos is not None:
                        self.update_view_camera()
                    fig.scene.render()
                    if self.paused:
                        self.capture_frame()
                elif received_info == 'new_sequence':
                    if self.snap_to_cam_on_scene_change:
                        self.initialized_camera = False
                        self.current_viewinfos = None
                    self.q_posedata.task_done()
                elif received_info == 'stop_visualization':
                    if self.q_out_video_frames is not None:
                        self.q_out_video_frames.put('stop_video_writing')
                    mlab.close(all=True)
                    self.q_posedata.task_done()
                    return
                else:
                    self.update_visu(received_info)
                    self.update_view_camera()
                    self.capture_frame()
                    self.q_posedata.task_done()
                    if self.step_one_by_one:
                        self.step_one_by_one = False
                        self.paused = True
                yield

        _ = anim()
        mlab.show()

    def update_visu(self, view_infos):
        import poseviz.draw2d
        pose_displayed_view_infos = (
            view_infos if self.pose_displayed_cam_id is None
            else [view_infos[self.pose_displayed_cam_id]])

        all_poses = [p for v in pose_displayed_view_infos for p in v.poses]
        all_poses_true = [p for v in pose_displayed_view_infos for p in v.poses_true]
        all_poses_alt = [p for v in pose_displayed_view_infos for p in v.poses_alt]

        self.current_viewinfos = view_infos

        if self.multicolor_detections:
            box_colors = poseviz.colors.cycle_over_colors(False)
        else:
            box_colors = itertools.repeat((31, 119, 180))
        joint_names, joint_edges = self.joint_info
        for view_info, viz in zip(view_infos, self.view_visualizers):
            for color, box in zip(box_colors, view_info.boxes):
                poseviz.draw2d.draw_box(view_info.frame, box, color, thickness=2)

            poses = all_poses if viz is self.view_visualizers[0] else None
            poses_alt = all_poses_alt if viz is self.view_visualizers[0] else None
            poses_true = all_poses_true if viz is self.view_visualizers[0] else None
            max_size = np.max(view_info.frame.shape[:2])
            if max_size < 512:
                thickness = 1
            elif max_size < 1024:
                thickness = 2
            else:
                thickness = 3

            if self.draw_2d_pose:
                pose_groups = [view_info.poses, view_info.poses_alt, view_info.poses_true]
                colors = [poseviz.colors.green, poseviz.colors.orange, poseviz.colors.red]
                for poses, color in zip(pose_groups, colors):
                    for pose in poses:
                        pose2d = view_info.camera.world_to_image(pose)
                        poseviz.draw2d.draw_stick_figure_2d_inplace(
                            view_info.frame, pose2d, joint_edges, thickness, color=color)
            viz.update(view_info.camera, view_info.frame, poses, poses_alt, poses_true)

    def update_view_camera(self):
        import poseviz.mayavi_util
        main_view_info = self.current_viewinfos[self.main_cam]
        if self.camera_type == 'original' or (
                self.camera_type == 'free' and not self.initialized_camera):
            poseviz.mayavi_util.set_view_to_camera(
                main_view_info.camera,
                image_size=(main_view_info.frame.shape[1], main_view_info.frame.shape[0]),
                allow_roll=False)
            self.initialized_camera = True

        if self.camera_type == 'bird' and not self.initialized_camera and len(
                main_view_info.poses) > 0:
            pivot = np.mean(main_view_info.poses, axis=(0, 1))
            camera_view = main_view_info.camera.copy()
            camera_view.t = (camera_view.t - pivot) * 1.5 + pivot
            camera_view.orbit_around(pivot, np.deg2rad(20), 'vertical')
            camera_view.orbit_around(pivot, np.deg2rad(-10), 'horizontal')
            poseviz.mayavi_util.set_view_to_camera(
                camera_view, pivot=pivot, image_size=
                (main_view_info.frame.shape[1], main_view_info.frame.shape[0]),
                allow_roll=False)
            self.initialized_camera = True

    def capture_frame(self):
        from mayavi import mlab
        if self.q_out_video_frames is not None:
            out_frame = mlab.screenshot(antialiased=True)
            # factor = 1 / 3
            # if self.camera_type != 'original' and False:
            #     image = main_view_info.frame
            #     image = np.clip(image.astype(np.float32) + 50, 0, 255).astype(np.uint8)
            #     for pred in main_view_info.poses:
            #         image = poseviz.draw2d.draw_stick_figure_2d(
            #             image, main_view_info.camera.world_to_image(pred), self.joint_info, 3)
            #     out_frame[:illust.shape[0], :illust.shape[1]] = illust
            self.q_out_video_frames.put(out_frame)

    def _on_keypress(self, obj, ev):
        key = obj.GetKeySym()
        if key == 'x':
            self.paused = not self.paused
        elif key == 't':
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = None

        if key == 'c':
            self.step_one_by_one = True
            self.paused = False
        else:
            try:
                self.main_cam = max(0, min(int(key) - 1, self.n_views - 1))
                self.initialized_camera = False
            except ValueError:
                pass


def viewinfo_tf_to_numpy(v: ViewInfo):
    return ViewInfo._make(map(tf_to_numpy, v))


def tf_to_numpy(x):
    try:
        import tensorflow as tf
    except ImportError:
        return x

    if isinstance(x, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        return x.numpy()
    return x


def _main_visualize(*args, **kwargs):
    viz = PoseVizMayaviSide(*args, **kwargs)
    viz.run_loop()
