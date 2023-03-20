import contextlib
import itertools
import pickle
import queue
from typing import List

import collections
import multiprocessing as mp
import numpy as np
import os

import poseviz.draw2d
import poseviz.colors
import poseviz.video_writing

ViewInfo = collections.namedtuple(
    'ViewInfo', ['frame', 'boxes', 'poses', 'camera', 'poses_true', 'poses_alt'],
    defaults=(None, (), (), None, (), ()))


class PoseViz:
    def __init__(
            self, joint_names, joint_edges, camera_type='free', n_views=1, world_up=(0, -1, 0),
            ground_plane_height=-1000, downscale=None, downscale_main=None,
            viz_fps=100, queue_size=64, draw_detections=True, multicolor_detections=False,
            snap_to_cam_on_scene_change=True, high_quality=True, draw_2d_pose=False,
            show_camera_wireframe=True, show_field_of_view=True, resolution=(1280, 720),
            use_virtual_display=False, show_virtual_display=True, show_ground_plane=True,
            paused=False, camera_view_padding=0.2):
        """The main class that creates a visualizer process, and provides an interface to update the visualization state.
        This class supports the Python **context manager** protocol to close the visualization at
        the end.

        Args:
            joint_names (iterable of strings): Names of the human body joints that will be
                visualized. Left joints must start with 'l', right joints with 'r', mid-body joints
                with something else. Currently only the first character is inspected in each joint
                name, to color the left, right and middle part of the stick figure differently.
            joint_edges (iterable of int pairs): joint index pairs, describing the
                bone-connectivity of the stick  figure.
            camera_type (string): One of 'original', 'free' or 'bird' (experimental). 'original'
                forces the  view-camera (i.e. from which we see the visualization) to stay or move
                where the displayed camera  goes, typically so that we can see the scene through the
                recording camera from which the video was  recorded. 'free' means that the
                view-camera is free to be moved around by the user. 'bird' (  experimental) tries to
                automatically move to a nice location to be able to see both the person and  the
                recording camera from a third-person perspective.
            n_views (int): The number of cameras that will be displayed.
                world_up (3-vector): A 3-vector, the up vector in the world coordinate system in
                which the poses will be specified.
            ground_plane_height (float): The vertical position of the ground plane in the world
                coordinate system, along the up-vector.
            downscale (int): Image downscaling factor for display, to speed up the
                visualization.
            viz_fps (int): Target frames-per-second of the visualization. If the updates come
                faster than  this, the visualizer will block and wait, to ensure that visualization does not
                happen faster than this FPS. Of course, if the speed of updates do not deliver this
                fps, the visualization will also be slower.
            queue_size (int): Size of the internal queue used to communicate with the
                visualizer process.
            draw_detections (bool): Whether to draw detection boxes on the images.
            multicolor_detections (bool): Whether to color each detection box with a different color.
                This is useful when tracking people with consistent person IDs, and has no effect
                if `draw_detections` is False.
            snap_to_cam_on_scene_change (bool): Whether to reinitialize the view camera to the
                original camera on each change of sequence (through a call to ```viz.reinit_camera_view()```).
            high_quality (bool): Whether to use high-resolution spheres and tubes for the
                skeletons (set to False for better speed).
            draw_2d_pose (bool): Whether to draw the 2D skeleton on the displayed camera image.
            show_camera_wireframe (bool): Whether to visualize each camera as a pyramid-like wireframe object.
            show_field_of_view (bool): Whether to visualize an extended pyramid shape indicating the
             field of view of the cameras. Recommended to turn off in multi-camera
                setups, as otherwise the visualization can get crowded.
            resolution ((int, int)): The resolution of the visualization window (width,
                height) pair.
            use_virtual_display (bool): Whether to use a virtual display for visualization.
                There may be two reasons to do this. First, to allow higher-resolution visualization
                than the screen resolution. Windows that are larger than the display screen can be
                difficult to create under certain GUI managers like GNOME. Second, this can be a way to do off-screen rendering.
            show_virtual_display (bool): Whether to show the virtual display or to hide it (
                off-screen rendering). This has no effect if ```use_virtual_display``` is False.
            show_ground_plane (bool): Whether to visualize a checkerboard ground plane.
            paused (bool): Whether to start the visualization in paused state.
            camera_view_padding (float): When viewing the scence from a visualized camera position,
                it is often useful to also see beyond the edges of the video frame. The ```cameera_view_padding```
                value adjusts what fraction of the frame size is applied as padding around it.
                Example with [```camera_view_padding=0```](/poseviz/images/padding_0.jpg) and [```camera_view_padding=0.2```](/poseviz/images/padding_0.2.jpg)
        """

        self.q_posedata = mp.JoinableQueue(queue_size)

        self.q_out_video_frames = mp.JoinableQueue(queue_size)
        self.video_writer_process = mp.Process(
            target=poseviz.video_writing.main_video_writer,
            args=(self.q_out_video_frames,))
        self.video_writer_process.start()

        self.downscale_main = downscale_main or downscale or (1 if high_quality else 2)
        self.downscale = downscale or 4
        self.pickle_path = None
        self.record = []
        self.snap_to_cam_on_scene_change = snap_to_cam_on_scene_change
        self.main_cam_value = mp.Value('i', 0)

        self.visualizer_process = mp.Process(
            target=_main_visualize, args=(
                self.q_posedata, self.q_out_video_frames, joint_names, joint_edges, camera_type,
                n_views, world_up, ground_plane_height, viz_fps, draw_detections,
                multicolor_detections, snap_to_cam_on_scene_change, high_quality, draw_2d_pose,
                show_camera_wireframe, show_field_of_view, resolution, use_virtual_display,
                show_virtual_display, show_ground_plane, self.main_cam_value, paused,
                camera_view_padding))
        self.visualizer_process.start()

        if resolution[1] > 720:
            input('Move the window to be partially outside the screen, then press Enter...')
            # Spent a lot of time trying to fix this, but it's very tricky and frustrating!
            # We have to manually drag the window partially off-screen, else it's impossible to set the size
            # larger than the display resolution! If you try, it will just snap to the image borders (get maximized).
            # An alternative would be like https://unix.stackexchange.com/a/680848/291533, ie
            # we can take the window away from the window manager and then freely set its size and position.
            # The downside is that we are no longer able to move the window with the mouse and it is stuck in the foreground.
            # So here we rather go the manual route.
            # Further info:
            # https://www.reddit.com/r/kde/comments/mzza4d/programmatically_moving_windows_off_the_screen/
            # https://unix.stackexchange.com/questions/517396/is-it-possible-to-move-window-out-of-the-screen-border#comment1396167_517396
            # https://github.com/jordansissel/xdotool/issues/186#issuecomment-470032261
            # Another option is some kind of virtual screen, off-screen rendering etc.
            # But then there are issues with hardware acceleration... It's quite complex.
            os.system('/usr/bin/xdotool search --name ^PoseViz$'
                      f' windowsize {resolution[0]} {resolution[1]}')

    def update(
            self, frame, boxes, poses, camera, poses_true=(), poses_alt=(), viz_camera=None,
            viz_imshape=None, block=True):
        """Update the visualization for a new timestep, assuming a single-camera setup.

        Args:
            frame (uint8): RGB image frame [H, W, 3], the image to be displayed on the camera.
            boxes (seq of np.ndarray): bounding boxes of the persons
            poses (seq of np.ndarray): the world
            camera (poseviz.Camera):
            poses_true (seq of np.ndarray):
            poses_alt (seq of np.ndarray):
            block (bool): decides what to do if the buffer queue is full because the visualizer
              is slower than the update calls. If true, the thread will block (wait). If false,
              the current update call is ignored (frame dropping).
        """
        viewinfo = ViewInfo(frame, boxes, poses, camera, poses_true, poses_alt)
        self.update_multiview([viewinfo], viz_camera, viz_imshape, block=block)

    def update_multiview(
            self, view_infos: List[ViewInfo], viz_camera=None, viz_imshape=None, block=True):
        """Update the visualization for a new timestep, with multi-view data.

        Args:
            view_infos: The view information for each view.
            block (bool): decides what to do if the buffer queue is full because the visualizer
              is slower than the update calls. If true, the thread will block (wait). If false,
              the current update call is ignored (frame dropping).
        """
        view_infos = list(map(viewinfo_tf_to_numpy, view_infos))
        if self.pickle_path is not None:
            recorded_view_infos = [
                ViewInfo(None, v.boxes, v.poses, v.camera, v.poses_true, v.poses_alt)
                for v in view_infos]
            self.record.append(recorded_view_infos)

        if self.downscale != 1 or self.downscale_main != 1:
            for i in range(len(view_infos)):
                v = view_infos[i]
                d = self.downscale_main if i == self.main_cam_value.value else self.downscale
                c = v.camera.copy()
                c.scale_output(1 / d)
                view_infos[i] = ViewInfo(
                    poseviz.draw2d.resize_by_factor(v.frame, 1 / d),
                    v.boxes / d, v.poses, c, v.poses_true, v.poses_alt)

        try:
            self.q_posedata.put((view_infos, viz_camera, viz_imshape), block=block)
        except queue.Full:
            pass

    def reinit_camera_view(self):
        """Waits until the current sequence has finished visualizing, then notifies PoseViz that
        a new sequence is now being visualized, so the view camera may need to be
        reinitialized.
        """
        self.q_posedata.put('reinit_camera_view')

    def new_sequence_output(self, new_video_path=None, fps=None, new_pickle_path=None):
        """Waits until the buffered information is visualized and then starts a new output video
        for the upcoming timesteps.

        Args:
            new_video_path (str): Path where the new video should be generated.
            fps (int): 'frames per second' setting of the new video to be generated
        """
        if new_video_path is not None:
            self.q_posedata.join()
            self.q_out_video_frames.put((new_video_path, fps))

        if self.pickle_path is not None:
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(self.record, f)
            self.record.clear()

        if new_pickle_path == '':
            noext, ext = os.path.splitext(new_video_path)
            self.pickle_path = noext + '.pkl'
        else:
            self.pickle_path = new_pickle_path

    def finalize_sequence_output(self):
        self.q_posedata.join()
        self.q_out_video_frames.put('close_current_video')
        self.q_out_video_frames.join()

    def close(self):
        """Closes the visualization process and window.
        """
        self.q_posedata.put('stop_visualization')  # close mayavi
        self._join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _join(self):
        self.q_posedata.join()
        self.q_out_video_frames.join()


class PoseVizMayaviSide:
    def __init__(
            self, q_posedata, q_out_video_frames, joint_names, joint_edges, camera_type, n_views,
            world_up, ground_plane_height, fps, draw_detections, multicolor_detections,
            snap_to_cam_on_scene_change, high_quality, draw_2d_pose, show_camera_wireframe,
            show_field_of_view, resolution, use_virtual_display, show_virtual_display,
            show_ground_plane, main_cam_value, paused, camera_view_padding):

        self.q_posedata = q_posedata
        self.q_out_video_frames = q_out_video_frames

        self.camera_type = camera_type
        self.joint_info = (joint_names, joint_edges)
        self.initialized_camera = False
        self.n_views = n_views
        self.world_up = world_up
        self.draw_detections = draw_detections
        self.multicolor_detections = multicolor_detections
        self.main_cam = 0
        self.pose_displayed_cam_id = None
        self.fps = fps
        self.snap_to_cam_on_scene_change = snap_to_cam_on_scene_change
        self.ground_plane_height = ground_plane_height
        self.show_ground_plane = show_ground_plane
        self.step_one_by_one = False
        self.paused = paused
        self.current_viewinfos = None
        self.high_quality = high_quality
        self.draw_2d_pose = draw_2d_pose
        self.show_camera_wireframe = show_camera_wireframe
        self.show_field_of_view = show_field_of_view
        self.resolution = resolution
        self.use_virtual_display = use_virtual_display
        self.show_virtual_display = show_virtual_display
        self.main_cam_value = main_cam_value
        self.camera_view_padding = camera_view_padding

    def run_loop(self):
        if self.use_virtual_display:
            import pyvirtualdisplay
            display = pyvirtualdisplay.Display(
                visible=self.show_virtual_display, size=self.resolution)
        else:
            display = contextlib.nullcontext()

        with display:
            # Imports are here so Mayavi is loaded in the visualizer process,
            # and potentially under the virtual display
            from mayavi import mlab
            import poseviz.init
            import poseviz.components.main_viz
            import poseviz.components.ground_plane_viz
            import poseviz.mayavi_util
            poseviz.mayavi_util.set_world_up(self.world_up)
            fig = poseviz.init.initialize(size=self.resolution)
            fig.scene.interactor.add_observer('KeyPressEvent', self._on_keypress)
            if self.show_ground_plane:
                poseviz.components.ground_plane_viz.draw_checkerboard(
                    ground_plane_height=self.ground_plane_height)
            self.view_visualizers = [poseviz.components.main_viz.MainViz(
                self.joint_info, self.joint_info, self.joint_info, self.camera_type,
                show_image=True, high_quality=self.high_quality,
                show_field_of_view=self.show_field_of_view,
                show_camera_wireframe=self.show_camera_wireframe)
                for _ in range(self.n_views)]
            delay = max(10, int(round(1000 / self.fps)))

            # Need to store it in a variable to make sure it's not
            # destructed by the garbage collector!
            _ = mlab.animate(delay=delay, ui=False)(self.animate)(fig)
            mlab.show()

    def animate(self, fig):
        # The main animation loop of the visualizer
        # The infinite loop pops an incoming command or frame information from the queue
        # and processes it. If there's no incoming information, we just render the scene and
        # continue to iterate.
        from mayavi import mlab

        while True:
            # Get a new command or frame visualization data from the multiprocessing queue
            received_info = self.receive_info()

            if received_info == 'nothing':
                if self.current_viewinfos is not None:
                    self.update_view_camera()
                fig.scene.render()
                if self.paused:
                    self.capture_frame()
            elif received_info == 'reinit_camera_view':
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
                viewinfos, viz_camera, viz_imshape = received_info
                self.update_visu(viewinfos)
                self.update_view_camera(viz_camera, viz_imshape)
                self.capture_frame()

                if self.step_one_by_one:
                    self.step_one_by_one = False
                    self.paused = True

                self.q_posedata.task_done()

            yield

    def receive_info(self):
        """Gets a new command or frame visualization data from the multiprocessing queue.
        Returns 'nothing' if the queue is empty or the visualization is paused."""
        if self.paused:
            return 'nothing'

        try:
            return self.q_posedata.get_nowait()
        except queue.Empty:
            return 'nothing'

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
        for i_viz, (view_info, viz) in enumerate(zip(view_infos, self.view_visualizers)):
            if self.draw_detections:
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

            viz.update(
                view_info.camera, view_info.frame, poses, poses_alt, poses_true,
                highlight=self.pose_displayed_cam_id == i_viz)

    def update_view_camera(self, camera=None, imshape=None):
        import poseviz.mayavi_util
        main_view_info = self.current_viewinfos[self.main_cam]

        if camera is not None:
            if imshape is None:
                imshape = main_view_info.frame.shape
            poseviz.mayavi_util.set_view_to_camera(
                camera, image_size=(imshape[1], imshape[0]), allow_roll=True,
                camera_view_padding=self.camera_view_padding)
        elif self.camera_type == 'original' or (
                self.camera_type == 'free' and not self.initialized_camera):
            poseviz.mayavi_util.set_view_to_camera(
                main_view_info.camera,
                image_size=(main_view_info.frame.shape[1], main_view_info.frame.shape[0]),
                allow_roll=False, camera_view_padding=self.camera_view_padding)
        elif self.camera_type == 'bird' and not self.initialized_camera:
            pivot = main_view_info.camera.t + main_view_info.camera.R[2] * 2000
            camera_view = main_view_info.camera.copy()
            camera_view.t = (camera_view.t - pivot) * 1.35 + pivot
            camera_view.orbit_around(pivot, np.deg2rad(-25), 'vertical')
            camera_view.orbit_around(pivot, np.deg2rad(-15), 'horizontal')
            poseviz.mayavi_util.set_view_to_camera(
                camera_view, pivot=pivot, view_angle=55, allow_roll=False,
                camera_view_padding=self.camera_view_padding)

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
            # Pause
            self.paused = not self.paused
        elif key == 't':
            # Toggle view poses predicted from all views or just one
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = None
        elif key == 'c':
            # Frame by frame stepping on key press
            self.step_one_by_one = True
            self.paused = False
        elif key == 'n':
            # Cycle the view camera
            self.main_cam = (self.main_cam + 1) % self.n_views
            self.main_cam_value.value = self.main_cam
            self.initialized_camera = False
        elif key == 'm':
            # Cycle which camera's predicted pose is displayed
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = (self.pose_displayed_cam_id + 1) % self.n_views
        elif key == 'o':
            # Cycle both the view camera and which camera's predicted pose is displayed
            self.main_cam = (self.main_cam + 1) % self.n_views
            self.main_cam_value.value = self.main_cam
            self.pose_displayed_cam_id = self.main_cam
            self.initialized_camera = False
        elif key in ('d', 'g'):
            # Snap to nearest camera
            import poseviz.mayavi_util
            display_cameras = [vi.camera for vi in self.current_viewinfos]
            viewing_camera = poseviz.mayavi_util.get_current_view_as_camera()
            self.main_cam = np.argmin(
                [np.linalg.norm(viewing_camera.t - c.t) for c in display_cameras])
            self.main_cam_value.value = self.main_cam
            self.initialized_camera = False
            if key == 'g':
                # Also display the predicted pose from this nearest camera
                self.pose_displayed_cam_id = self.main_cam
        elif key == 'u':
            # Show just the main cam pred
            self.pose_displayed_cam_id = self.main_cam
        else:
            try:
                self.main_cam = max(0, min(int(key) - 1, self.n_views - 1))
                self.main_cam_value.value = self.main_cam
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
