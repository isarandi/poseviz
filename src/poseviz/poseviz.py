import contextlib
import ctypes
import itertools
import multiprocessing as mp
import multiprocessing.shared_memory
import multiprocessing.sharedctypes
import os
import queue
import signal
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Optional, TYPE_CHECKING

import cameravision
import numpy as np
import poseviz.colors
import poseviz.draw2d
import poseviz.view_info
import simplepyutils as spu
import framepump
from poseviz import messages

if TYPE_CHECKING:
    from poseviz import ViewInfo
else:
    from poseviz.view_info import ViewInfo

# We must not globally import mlab, because then it will be imported in the main process
# and the new process will not be able to open a mayavi window.
# This means, we also cannot import various modules like poseviz.components.main_viz
# We import these in the methods of PoseVizMayaviSide.


class PoseViz(AbstractContextManager):
    def __init__(
        self,
        joint_names=None,
        joint_edges=None,
        camera_type="free",
        n_views=1,
        world_up=(0, -1, 0),
        ground_plane_height=-1000,
        downscale=None,
        downscale_main=None,
        viz_fps=100,
        queue_size=64,
        draw_detections=True,
        multicolor_detections=False,
        snap_to_cam_on_scene_change=True,
        high_quality=True,
        draw_2d_pose=False,
        show_camera_wireframe=True,
        show_field_of_view=True,
        resolution=(1280, 720),
        use_virtual_display=False,
        show_virtual_display=True,
        show_ground_plane=True,
        paused=False,
        camera_view_padding=0.2,
        body_model_faces=None,
        show_image=True,
        out_video_path=None,
        out_fps=None,
        audio_path=None,
        max_pixels_per_frame=1920 * 1080,
    ):
        """The main class that creates a visualizer process, and provides an interface to update
        the visualization state.
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
                faster than  this, the visualizer will block and wait, to ensure that
                visualization does not
                happen faster than this FPS. Of course, if the speed of updates do not deliver this
                fps, the visualization will also be slower.
            queue_size (int): Size of the internal queue used to communicate with the
                visualizer process.
            draw_detections (bool): Whether to draw detection boxes on the images.
            multicolor_detections (bool): Whether to color each detection box with a different
            color.
                This is useful when tracking people with consistent person IDs, and has no effect
                if `draw_detections` is False.
            snap_to_cam_on_scene_change (bool): Whether to reinitialize the view camera to the
                original camera on each change of sequence (through a call to
                ```viz.reinit_camera_view()```).
            high_quality (bool): Whether to use high-resolution spheres and tubes for the
                skeletons (set to False for better speed).
            draw_2d_pose (bool): Whether to draw the 2D skeleton on the displayed camera image.
            show_camera_wireframe (bool): Whether to visualize each camera as a pyramid-like
            wireframe object.
            show_field_of_view (bool): Whether to visualize an extended pyramid shape indicating the
             field of view of the cameras. Recommended to turn off in multi-camera
                setups, as otherwise the visualization can get crowded.
            resolution ((int, int)): The resolution of the visualization window (width,
                height) pair.
            use_virtual_display (bool): Whether to use a virtual display for visualization.
                There may be two reasons to do this. First, to allow higher-resolution visualization
                than the screen resolution. Windows that are larger than the display screen can be
                difficult to create under certain GUI managers like GNOME. Second, this can be a
                way to do off-screen rendering.
            show_virtual_display (bool): Whether to show the virtual display or to hide it (
                off-screen rendering). This has no effect if ```use_virtual_display``` is False.
            show_ground_plane (bool): Whether to visualize a checkerboard ground plane.
            paused (bool): Whether to start the visualization in paused state.
            camera_view_padding (float): When viewing the scence from a visualized camera position,
                it is often useful to also see beyond the edges of the video frame. The
                ```cameera_view_padding```
                value adjusts what fraction of the frame size is applied as padding around it.
                Example with [```camera_view_padding=0```](/poseviz/images/padding_0.jpg) and [
                ```camera_view_padding=0.2```](/poseviz/images/padding_0.2.jpg)
            out_video_path: Path where the output video should be generated. (It can also be started
                later with ```new_sequence_output```). If None, no video will be generated for now.
            out_fps: 'frames per second' setting of the output video to be generated.
            audio_path: Path to the audio source file, which may also be a video file whose audio
                will be copied over to the output video.
        """

        n_threads_undist = 12
        queue_size_undist = min(max(16, n_views), 2 * n_views)
        queue_size_post = 2
        queue_size_waiter = queue_size
        self.q_messages_pre = queue.Queue(queue_size_waiter)
        if paused:
            self.pause()
        self.q_messages_post = mp.JoinableQueue(queue_size_post)

        self.undistort_pool = spu.ThrottledPool(
            n_threads_undist, use_threads=True, task_buffer_size=queue_size_undist
        )

        self.ringbuffer_size = queue_size_undist + queue_size_waiter + queue_size_post + 1
        self.raw_arrays = [
            multiprocessing.sharedctypes.RawArray(
                ctypes.c_uint8, self.ringbuffer_size * max_pixels_per_frame * 3
            )
            for _ in range(n_views)
        ]
        self.ring_index = 0

        self.posedata_waiter_thread = threading.Thread(
            target=main_posedata_waiter,
            args=(self.q_messages_pre, self.q_messages_post),
            daemon=True,
        )
        self.posedata_waiter_thread.start()

        self.downscale_main = downscale_main or downscale or (1 if high_quality else 2)
        self.downscale = downscale or 4
        self.snap_to_cam_on_scene_change = snap_to_cam_on_scene_change
        self.main_cam_value = mp.Value("i", 0)

        self.visualizer_process = mp.Process(
            target=_main_visualize,
            args=(
                self.q_messages_post,
                joint_names,
                joint_edges,
                camera_type,
                n_views,
                world_up,
                ground_plane_height,
                viz_fps,
                draw_detections,
                multicolor_detections,
                snap_to_cam_on_scene_change,
                high_quality,
                draw_2d_pose,
                show_camera_wireframe,
                show_field_of_view,
                resolution,
                use_virtual_display,
                show_virtual_display,
                show_ground_plane,
                self.main_cam_value,
                camera_view_padding,
                body_model_faces,
                show_image,
                self.raw_arrays,
            ),
            daemon=True,
        )
        self.visualizer_process.start()

        if resolution[1] > 1080:
            input("Move the window to be partially outside the screen, then press Enter...")
            # Spent a lot of time trying to fix this, but it's very tricky and frustrating!
            # We have to manually drag the window partially off-screen, else it's impossible to
            # set the size
            # larger than the display resolution! If you try, it will just snap to the screen
            # borders (get maximized).
            # An alternative would be like https://unix.stackexchange.com/a/680848/291533, ie
            # we can take the window away from the window manager and then freely set its size
            # and position.
            # The downside is that we are no longer able to move the window with the mouse and it
            # is stuck in the foreground.
            # So here we rather go the manual route.
            # Further info:
            # https://www.reddit.com/r/kde/comments/mzza4d
            # /programmatically_moving_windows_off_the_screen/
            # https://unix.stackexchange.com/questions/517396/is-it-possible-to-move-window-out
            # -of-the-screen-border#comment1396167_517396
            # https://github.com/jordansissel/xdotool/issues/186#issuecomment-470032261
            # Another option is some kind of virtual screen, off-screen rendering etc.
            # But then there are issues with hardware acceleration... It's quite complex.
            os.system(
                "/usr/bin/xdotool search --name ^PoseViz$"
                f" windowsize {resolution[0]} {resolution[1]}"
            )

        if out_video_path is not None:
            self.new_sequence_output(out_video_path, out_fps, audio_path)

    def update(
        self,
        frame: np.ndarray,
        boxes=(),
        poses=(),
        camera: Optional[cameravision.Camera] = None,
        poses_true=(),
        poses_alt=(),
        vertices=(),
        vertices_true=(),
        vertices_alt=(),
        viz_camera=None,
        viz_imshape=None,
        block=True,
        uncerts=None,
        uncerts_alt=None,
    ):
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
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), np.float32)

        if uncerts is not None:
            vertices = [
                np.concatenate([v, uncert[..., np.newaxis]], axis=-1)
                for v, uncert in zip(vertices, uncerts)
            ]
        if uncerts_alt is not None:
            vertices_alt = [
                np.concatenate([v, uncert[..., np.newaxis]], axis=-1)
                for v, uncert in zip(vertices_alt, uncerts_alt)
            ]

        viewinfo = ViewInfo(
            frame,
            boxes,
            poses,
            camera,
            poses_true,
            poses_alt,
            vertices,
            vertices_true,
            vertices_alt,
        )
        self.update_multiview([viewinfo], viz_camera, viz_imshape, block=block)

    def update_multiview(
        self,
        view_infos: Sequence[ViewInfo],
        viz_camera: Optional[cameravision.Camera] = None,
        viz_imshape=None,
        block: bool = True,
    ):
        """Update the visualization for a new timestep, with multi-view data.

        Args:
            view_infos: The view information for each view.
            viz_camera: The camera to be used for the visualization.
            viz_imshape: The shape of the image of the visualization camera.
            block: decides what to do if the buffer queue is full because the visualizer
              is slower than the update calls. If true, the thread will block (wait). If false,
              the current update call is ignored (frame dropping).
        """

        for i in range(len(view_infos)):
            if len(view_infos[i].boxes) == 0:
                view_infos[i].boxes = np.zeros((0, 4), np.float32)

            d = self.downscale_main if i == self.main_cam_value.value else self.downscale
            if view_infos[i].frame is not None:
                new_imshape = poseviz.draw2d.rounded_int_tuple(
                    np.array(view_infos[i].frame.shape[:2], np.float32) / d
                )
                dst = np_from_raw_array(
                    self.raw_arrays[i],
                    (new_imshape[0], new_imshape[1], 3),
                    self.ring_index,
                    np.uint8,
                )
                view_infos[i] = self.undistort_pool.apply_async(
                    poseviz.view_info.downscale_and_undistort_view_info,
                    (view_infos[i], dst, self.ring_index),
                )
            else:
                view_infos[i] = self.undistort_pool.apply_async(lambda x: x, (view_infos[i],))
        self.ring_index = (self.ring_index + 1) % self.ringbuffer_size

        try:
            self.q_messages_pre.put(
                messages.UpdateScene(
                    view_infos=view_infos, viz_camera=viz_camera, viz_imshape=viz_imshape
                ),
                block=block,
            )
        except queue.Full:
            pass

    def reinit_camera_view(self):
        """Waits until the current sequence has finished visualizing, then notifies PoseViz that
        a new sequence is now being visualized, so the view camera may need to be
        reinitialized.
        """
        self.q_messages_pre.put(messages.ReinitCameraView())

    def pause(self):
        self.q_messages_pre.put(messages.Pause())

    def resume(self):
        self.q_messages_pre.put(messages.Resume())

    def new_sequence_output(
        self,
        new_video_path: Optional[str] = None,
        fps: Optional[int] = None,
        new_camera_trajectory_path: Optional[str] = None,
        audio_source_path: Optional[str] = None,
    ):
        """Waits until the buffered information is visualized and then starts a new output video
        for the upcoming timesteps.

        Args:
            new_video_path: Path where the new video should be generated.
            fps: 'frames per second' setting of the new video to be generated
            new_camera_trajectory_path: Path where the camera trajectory should be saved.
        """
        if new_video_path is not None or new_camera_trajectory_path is not None:
            self.q_messages_pre.put(
                messages.StartSequence(
                    video_path=new_video_path,
                    fps=fps,
                    camera_trajectory_path=new_camera_trajectory_path,
                    audio_source_path=audio_source_path,
                )
            )

    def finalize_sequence_output(self):
        """Notifies PoseViz that the current sequence is finished and the output video should be
        finalized."""
        self.q_messages_pre.put(messages.EndSequence())

    def close(self):
        """Closes the visualization process and window."""
        self.undistort_pool.finish()
        self.q_messages_pre.put(messages.Quit())
        self._join()
        self.posedata_waiter_thread.join()
        self.visualizer_process.join()

    def _join(self):
        self.undistort_pool.join()
        self.q_messages_pre.join()
        self.q_messages_post.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager protocol."""
        self.close()


def np_from_raw_array(raw_array, elem_shape, index, dtype):
    return np.frombuffer(
        buffer=raw_array,
        dtype=dtype,
        count=np.prod(elem_shape),
        offset=index * np.prod(elem_shape) * np.dtype(dtype).itemsize,
    ).reshape(elem_shape)


class PoseVizMayaviSide:
    def __init__(
        self,
        q_messages,
        joint_names,
        joint_edges,
        camera_type,
        n_views,
        world_up,
        ground_plane_height,
        fps,
        draw_detections,
        multicolor_detections,
        snap_to_cam_on_scene_change,
        high_quality,
        draw_2d_pose,
        show_camera_wireframe,
        show_field_of_view,
        resolution,
        use_virtual_display,
        show_virtual_display,
        show_ground_plane,
        main_cam_value,
        camera_view_padding,
        body_model_faces,
        show_image,
        raw_arrays,
    ):
        self.q_messages = q_messages
        self.video_writer = framepump.VideoWriter()

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
        self.paused = False
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
        self.body_model_faces = body_model_faces
        self.i_pred_frame = 0
        self.camera_trajectory = []
        self.camera_trajectory_path = None
        self.show_image = show_image
        self.raw_arrays = raw_arrays

    def run_loop(self):
        if self.use_virtual_display:
            import pyvirtualdisplay

            display = pyvirtualdisplay.Display(
                visible=self.show_virtual_display, size=self.resolution
            )
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
            fig = poseviz.init.initialize(self.resolution)
            fig.scene.interactor.add_observer("KeyPressEvent", self._on_keypress)
            if self.show_ground_plane:
                poseviz.components.ground_plane_viz.draw_checkerboard(
                    ground_plane_height=self.ground_plane_height
                )
            self.view_visualizers = [
                poseviz.components.main_viz.MainViz(
                    self.joint_info,
                    self.joint_info,
                    self.joint_info,
                    self.camera_type,
                    show_image=self.show_image,
                    high_quality=self.high_quality,
                    show_field_of_view=self.show_field_of_view,
                    show_camera_wireframe=self.show_camera_wireframe,
                    body_model_faces=self.body_model_faces,
                )
                for _ in range(self.n_views)
            ]
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
            # Get a new message (command) from the multiprocessing queue
            msg = self.receive_message()
            if isinstance(msg, messages.Nothing):
                if self.current_viewinfos is not None:
                    self.update_view_camera()
                fig.scene.render()
                if self.paused:
                    self.capture_frame()
                yield
            elif isinstance(msg, messages.UpdateScene):
                self.update_visu(msg.view_infos)
                self.update_view_camera(msg.viz_camera, msg.viz_imshape)
                self.capture_frame()

                if self.step_one_by_one:
                    self.step_one_by_one = False
                    self.paused = True

                self.i_pred_frame += 1
                self.q_messages.task_done()
                yield
            elif isinstance(msg, messages.ReinitCameraView):
                if self.snap_to_cam_on_scene_change:
                    self.initialized_camera = False
                    self.current_viewinfos = None
                self.q_messages.task_done()
            elif isinstance(msg, messages.StartSequence):
                if msg.video_path is not None:
                    self.video_writer.start_sequence(
                        video_path=msg.video_path,
                        fps=msg.fps,
                        audio_source_path=msg.audio_source_path,
                    )
                self.camera_trajectory_path = msg.camera_trajectory_path
                self.i_pred_frame = 0

                # os.system('/usr/bin/xdotool search --name ^PoseViz$'
                #          f' windowsize --sync {msg.resolution[0]} {msg.resolution[1]}')
                self.q_messages.task_done()
                # yield
            elif isinstance(msg, messages.Pause):
                self.paused = True
                self.q_messages.task_done()
            elif isinstance(msg, messages.Resume):
                self.paused = False
                self.q_messages.task_done()
            elif isinstance(msg, messages.EndSequence):
                if self.camera_trajectory_path is not None:
                    spu.dump_pickle(self.camera_trajectory, self.camera_trajectory_path)
                self.video_writer.end_sequence()
                self.q_messages.task_done()
            elif isinstance(msg, messages.Quit):
                if self.camera_trajectory_path is not None:
                    spu.dump_pickle(self.camera_trajectory, self.camera_trajectory_path)
                mlab.close(all=True)
                self.video_writer.close()
                self.q_messages.task_done()
                return
            else:
                raise ValueError("Unknown message:", msg)

    def receive_message(self):
        """Gets a new command or frame visualization data from the multiprocessing queue.
        Returns messages.Nothing if the queue is empty or the visualization is paused."""
        if self.paused:
            return messages.Nothing()

        try:
            return self.q_messages.get_nowait()
        except queue.Empty:
            return messages.Nothing()

    def update_visu(self, view_infos):
        self.update_num_views(len(view_infos))  # Update the number of views if necessary

        for i, v in enumerate(view_infos):
            if v.frame is not None:
                shape, dtype, index = v.frame
                v.frame = np_from_raw_array(self.raw_arrays[i], shape, index, dtype)

        pose_displayed_view_infos = (
            view_infos
            if self.pose_displayed_cam_id is None
            else [view_infos[self.pose_displayed_cam_id]]
        )

        all_poses = [p for v in pose_displayed_view_infos for p in v.poses]
        all_poses_true = [p for v in pose_displayed_view_infos for p in v.poses_true]
        all_poses_alt = [p for v in pose_displayed_view_infos for p in v.poses_alt]
        all_vertices = [p for v in pose_displayed_view_infos for p in v.vertices]
        all_vertices_true = [p for v in pose_displayed_view_infos for p in v.vertices_true]
        all_vertices_alt = [p for v in pose_displayed_view_infos for p in v.vertices_alt]

        self.current_viewinfos = view_infos

        if self.multicolor_detections:
            box_colors = poseviz.colors.cycle_over_colors(False)
        else:
            box_colors = itertools.repeat((31, 119, 180))
        joint_names, joint_edges = self.joint_info
        for i_viz, (view_info, viz) in enumerate(zip(view_infos, self.view_visualizers)):
            poses = all_poses if viz is self.view_visualizers[0] else None
            poses_true = all_poses_true if viz is self.view_visualizers[0] else None
            poses_alt = all_poses_alt if viz is self.view_visualizers[0] else None
            vertices = all_vertices if viz is self.view_visualizers[0] else None
            vertices_true = all_vertices_true if viz is self.view_visualizers[0] else None
            vertices_alt = all_vertices_alt if viz is self.view_visualizers[0] else None

            max_size = np.max(view_info.frame.shape[:2])
            if max_size < 512:
                thickness = 1
            elif max_size < 1024:
                thickness = 2
            else:
                thickness = 3

            if self.draw_detections:
                for color, box in zip(box_colors, view_info.boxes):
                    poseviz.draw2d.draw_box(view_info.frame, box, color, thickness=thickness)

            if self.draw_2d_pose:
                pose_groups = [view_info.poses, view_info.poses_true, view_info.poses_alt]
                colors = [poseviz.colors.green, poseviz.colors.red, poseviz.colors.orange]
                for poses, color in zip(pose_groups, colors):
                    for pose in poses:
                        pose2d = view_info.camera.world_to_image(pose)
                        poseviz.draw2d.draw_stick_figure_2d_inplace(
                            view_info.frame, pose2d, joint_edges, thickness, color=color
                        )

            viz.update(
                view_info.camera,
                view_info.frame,
                poses,
                poses_true,
                poses_alt,
                vertices,
                vertices_true,
                vertices_alt,
                highlight=self.pose_displayed_cam_id == i_viz,
            )

    def update_num_views(self, new_n_views):
        if new_n_views > self.n_views:
            self.view_visualizers += [
                poseviz.components.main_viz.MainViz(
                    self.joint_info,
                    self.joint_info,
                    self.joint_info,
                    self.camera_type,
                    show_image=True,
                    high_quality=self.high_quality,
                    show_field_of_view=self.show_field_of_view,
                    show_camera_wireframe=self.show_camera_wireframe,
                    body_model_faces=self.body_model_faces,
                )
                for _ in range(new_n_views - self.n_views)
            ]
            self.n_views = new_n_views
        elif new_n_views < self.n_views:
            for viz in self.view_visualizers[new_n_views:]:
                viz.remove()
            del self.view_visualizers[new_n_views:]
            self.n_views = new_n_views

    def update_view_camera(self, camera=None, imshape=None):
        import poseviz.mayavi_util

        main_view_info = self.current_viewinfos[self.main_cam]

        if camera is not None:
            if imshape is None:
                imshape = main_view_info.frame.shape
            poseviz.mayavi_util.set_view_to_camera(
                camera, image_size=(imshape[1], imshape[0]), allow_roll=True, camera_view_padding=0
            )
        elif self.camera_type == "original" or (
            self.camera_type == "free" and not self.initialized_camera
        ):
            poseviz.mayavi_util.set_view_to_camera(
                main_view_info.camera,
                image_size=(main_view_info.frame.shape[1], main_view_info.frame.shape[0]),
                allow_roll=False,
                camera_view_padding=self.camera_view_padding,
            )
        elif self.camera_type == "bird" and not self.initialized_camera:
            pivot = main_view_info.camera.t + main_view_info.camera.R[2] * 2000
            camera_view = main_view_info.camera.copy()
            camera_view.t = (camera_view.t - pivot) * 1.35 + pivot
            camera_view.orbit_around(pivot, np.deg2rad(-25), "vertical")
            camera_view.orbit_around(pivot, np.deg2rad(-15), "horizontal")
            poseviz.mayavi_util.set_view_to_camera(
                camera_view,
                pivot=pivot,
                view_angle=55,
                allow_roll=False,
                camera_view_padding=self.camera_view_padding,
            )

        self.initialized_camera = True

    def capture_frame(self):
        from mayavi import mlab

        if self.camera_trajectory_path is not None:
            from poseviz.mayavi_util import get_current_view_as_camera

            viz_cam = get_current_view_as_camera()
            self.camera_trajectory.append((self.i_pred_frame, viz_cam))

        if self.video_writer.is_active():
            out_frame = mlab.screenshot(antialiased=False)
            # fig = mlab.gcf()
            # fig.scene.disable_render = True
            # fig.scene.off_screen_rendering = True
            # mlab.savefig(self.temp_image_path, size=self.resolution)
            # fig.scene.off_screen_rendering = False
            # fig.scene.disable_render = False
            # out_frame = imageio.imread(self.temp_image_path)
            # factor = 1 / 3
            # if self.camera_type != 'original' and False:
            #     image = main_view_info.frame
            #     image = np.clip(image.astype(np.float32) + 50, 0, 255).astype(np.uint8)
            #     for pred in main_view_info.poses:
            #         image = poseviz.draw2d.draw_stick_figure_2d(
            #             image, main_view_info.camera.world_to_image(pred), self.joint_info, 3)
            #     out_frame[:illust.shape[0], :illust.shape[1]] = illust
            self.video_writer.append_data(out_frame)

    def _on_keypress(self, obj, ev):
        key = obj.GetKeySym()
        if key == "x":
            # Pause
            self.paused = not self.paused
        elif key == "t":
            # Toggle view poses predicted from all views or just one
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = None
        elif key == "c":
            # Frame by frame stepping on key press
            self.step_one_by_one = True
            self.paused = False
        elif key == "n":
            # Cycle the view camera
            self.main_cam = (self.main_cam + 1) % self.n_views
            self.main_cam_value.value = self.main_cam
            self.initialized_camera = False
        elif key == "m":
            # Cycle which camera's predicted pose is displayed
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = (self.pose_displayed_cam_id + 1) % self.n_views
        elif key == "o":
            # Cycle both the view camera and which camera's predicted pose is displayed
            self.main_cam = (self.main_cam + 1) % self.n_views
            self.main_cam_value.value = self.main_cam
            self.pose_displayed_cam_id = self.main_cam
            self.initialized_camera = False
        elif key in ("d", "g"):
            # Snap to nearest camera
            import poseviz.mayavi_util

            display_cameras = [vi.camera for vi in self.current_viewinfos]
            viewing_camera = poseviz.mayavi_util.get_current_view_as_camera()
            self.main_cam = np.argmin(
                [np.linalg.norm(viewing_camera.t - c.t) for c in display_cameras]
            )
            self.main_cam_value.value = self.main_cam
            self.initialized_camera = False
            if key == "g":
                # Also display the predicted pose from this nearest camera
                self.pose_displayed_cam_id = self.main_cam
        elif key == "u":
            # Show just the main cam pred
            self.pose_displayed_cam_id = self.main_cam
        elif key == "z":
            # Show just the main cam pred
            self.camera_type = "original" if self.camera_type != "original" else "free"
            self.initialized_camera = False
        else:
            try:
                self.main_cam = max(0, min(int(key) - 1, self.n_views - 1))
                self.main_cam_value.value = self.main_cam
                self.initialized_camera = False
            except ValueError:
                pass


def _main_visualize(*args, **kwargs):
    _terminate_on_parent_death()
    remove_qt_paths_added_by_opencv()
    monkey_patched_numpy_types()
    viz = PoseVizMayaviSide(*args, **kwargs)
    viz.run_loop()


def _terminate_on_parent_death():
    # Make sure the visualizer process is terminated when the parent process dies, for whatever reason
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

    # ignore sigterm and sigint
    # without this, a single CTRL+C leaves the process hanging
    #signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def monkey_patched_numpy_types():
    """VTK tries to import np.bool etc which are not available anymore."""
    for name in ["bool", "int", "float", "complex", "object", "str"]:
        if name not in dir(np):
            setattr(np, name, getattr(np, name + "_"))


def remove_qt_paths_added_by_opencv():
    """Remove Qt paths added by OpenCV, which may cause conflicts with Qt as used by Mayavi."""
    # See also https://forum.qt.io/post/654289
    import sys

    # noinspection PyUnresolvedReferences
    import cv2

    try:
        from cv2.version import ci_build, headless

        ci_and_not_headless = ci_build and not headless
    except:
        ci_and_not_headless = False

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
        os.environ.pop("QT_QPA_FONTDIR")


def main_posedata_waiter(q_posedata_pre, q_posedata_post):
    while True:
        msg = q_posedata_pre.get()
        if isinstance(msg, messages.UpdateScene):
            msg.view_infos = [v.get() for v in msg.view_infos]

        q_posedata_post.put(msg)
        q_posedata_pre.task_done()
        if isinstance(msg, messages.Quit):
            return
