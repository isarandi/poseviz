import ctypes
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Optional, TYPE_CHECKING

import deltacamera
import numpy as np
import poseviz.colors
import poseviz.draw2d
import poseviz.view_info
import simplepyutils as spu
from poseviz import messages
from poseviz.shared_ring_buffer import SharedRingBuffer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from poseviz import ViewInfo
else:
    from poseviz.view_info import ViewInfo

# Use spawn context to avoid CUDA fork issues (CUDA initialized in parent breaks forked children)
_mp_ctx = mp.get_context("spawn")

try:
    import torch.multiprocessing  # registers CUDA IPC pickle reducers
except ImportError:
    pass


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
        render_resolution=None,  # If set, renders at this resolution (can be larger than display)
        use_virtual_display=False,
        show_virtual_display=True,
        show_ground_plane=True,
        paused=False,
        camera_view_padding=0.2,
        body_model_faces=None,
        show_image=True,
        image_plane_distance=1000,
        out_video_path=None,
        out_fps=None,
        audio_path=None,
        max_pixels_per_frame=1920 * 1080,
        flying_mode="camera",
        headless=None,
        gpu_encode=True,
        fullscreen=False,
        gpu_frames=False,
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
            n_views (int): Initial number of camera views to allocate. The actual count
                adjusts dynamically based on how many ViewInfos are passed to
                ``update_multiview``.
            world_up (3-vector): The up vector in the world coordinate system in
                which the poses will be specified.
            ground_plane_height (float): The vertical position of the ground plane in the world
                coordinate system, along the up-vector.
            downscale (int): Image downscaling factor for non-main cameras. Defaults to 4.
            downscale_main (int): Image downscaling factor for the main camera. Falls back
                to ``downscale`` if set, otherwise 1 when ``high_quality`` is True or
                2 when False.
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
            render_resolution ((int, int)): The resolution to render at, which can be larger
                than the window for high-resolution video output. If None, matches
                ``resolution``.
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
                ```camera_view_padding```
                value adjusts what fraction of the frame size is applied as padding around it.
                Example with [```camera_view_padding=0```](/poseviz/images/padding_0.jpg) and [
                ```camera_view_padding=0.2```](/poseviz/images/padding_0.2.jpg)
            body_model_faces: Face array for the body mesh model (e.g., SMPL), shape (F, 3).
                If None, mesh visualization is disabled.
            show_image (bool): Whether to show the camera image as a textured quad in the
                3D scene.
            image_plane_distance (float): Distance in mm from the camera origin at which the
                image quad is placed.
            out_video_path: Path where the output video should be generated. (It can also be started
                later with ```new_sequence_output```). If None, no video will be generated for now.
            out_fps: 'frames per second' setting of the output video to be generated.
            audio_path: Path to the audio source file, which may also be a video file whose audio
                will be copied over to the output video.
            max_pixels_per_frame (int): Maximum number of pixels per frame for shared memory
                ring buffer allocation. Only used when ``gpu_frames`` is False.
            flying_mode (str): Controls arrow-key flying direction. ``'camera'``: forward/back
                follow the camera's look direction (tilted down = fly down).
                ``'horizontal'``: forward/back stay in the horizontal plane regardless of
                where the camera is looking.
            headless: Whether to render without a visible window. If None (default),
                auto-detected from the environment: headless if neither DISPLAY nor
                WAYLAND_DISPLAY is set, otherwise windowed.
            gpu_encode (bool): Whether to use GPU-accelerated video encoding (NVENC). If
                False, falls back to CPU encoding.
            fullscreen (bool): Whether to start the window in fullscreen mode.
            gpu_frames (bool): Whether frames are GPU tensors instead of NumPy arrays.
                Accepts PyTorch CUDA tensors or any DLPack-compatible object (CuPy, JAX,
                etc.). Must be set at init time as it determines the frame transfer
                mechanism (CUDA IPC instead of shared memory ring buffers).
                Enables GPU-side image downscaling and undistortion.
        """

        if headless is None:
            headless = not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY')

        self.gpu_frames = gpu_frames
        queue_size_post = 6
        queue_size_waiter = queue_size
        self.q_messages_pre = queue.Queue(queue_size_waiter)
        if paused:
            self.pause()
        self.q_messages_post = _mp_ctx.JoinableQueue(queue_size_post)

        if gpu_frames:
            self.frame_rings = []
        else:
            n_threads_undist = 12
            queue_size_undist = min(max(16, n_views), 2 * n_views)
            self.undistort_pool = spu.ThrottledPool(
                n_threads_undist, use_threads=True, task_buffer_size=queue_size_undist
            )
            self.ringbuffer_size = (
                queue_size_undist + queue_size_waiter + queue_size_post + 1
            )
            self.frame_rings = [
                SharedRingBuffer(self.ringbuffer_size, max_pixels_per_frame * 3, np.uint8)
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
        self.main_cam_value = _mp_ctx.Value("i", 0)

        self.visualizer_process = _mp_ctx.Process(
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
                render_resolution,  # None means "match display resolution"
                use_virtual_display,
                show_virtual_display,
                show_ground_plane,
                self.main_cam_value,
                camera_view_padding,
                body_model_faces,
                show_image,
                image_plane_distance,
                self.frame_rings,
                flying_mode,
                headless,
                gpu_encode,
                fullscreen,
            ),
            daemon=True,
        )
        self.visualizer_process.start()

        # NOTE: The >1080p workaround below shouldn't be needed since
        # the GL renderer handles window sizing properly via GLFW.
        # if resolution[1] > 1080:
        #     input("Move the window to be partially outside the screen, then press Enter...")
        #     os.system(
        #         "/usr/bin/xdotool search --name ^PoseViz$"
        #         f" windowsize {resolution[0]} {resolution[1]}"
        #     )

        if out_video_path is not None:
            self.new_sequence_output(out_video_path, out_fps, audio_path)

    def update(
        self,
        frame: np.ndarray,
        boxes=(),
        poses=(),
        camera: Optional[deltacamera.Camera] = None,
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
        viz_camera: Optional[deltacamera.Camera] = None,
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

        # Check if we can enqueue before doing any work (non-blocking mode)
        if not block and self.q_messages_pre.full():
            return

        for i in range(len(view_infos)):
            if len(view_infos[i].boxes) == 0:
                view_infos[i].boxes = np.zeros((0, 4), np.float32)

            d = (
                self.downscale_main
                if i == self.main_cam_value.value
                else self.downscale
            )
            if self.gpu_frames:
                if view_infos[i].frame is not None:
                    view_infos[i] = _ImmediateResult(
                        poseviz.view_info.gpu_downscale_and_undistort(view_infos[i], d)
                    )
                else:
                    view_infos[i] = _ImmediateResult(view_infos[i])
            else:
                if view_infos[i].frame is not None:
                    new_imshape = poseviz.draw2d.rounded_int_tuple(
                        np.array(view_infos[i].frame.shape[:2], np.float32) / d
                    )
                    n_pixels = new_imshape[0] * new_imshape[1]
                    max_pixels = self.frame_rings[i].max_elems // 3
                    if n_pixels > max_pixels:
                        raise ValueError(
                            f'Downscaled frame ({new_imshape[1]}x{new_imshape[0]} = '
                            f'{n_pixels} pixels) exceeds max_pixels_per_frame '
                            f'({max_pixels}). Pass a larger max_pixels_per_frame to '
                            f'PoseViz or increase the downscale factor.'
                        )
                    dst = self.frame_rings[i].get_slot(
                        self.ring_index, (new_imshape[0], new_imshape[1], 3)
                    )
                    view_infos[i] = self.undistort_pool.apply_async(
                        poseviz.view_info.downscale_and_undistort_view_info,
                        (view_infos[i], dst, self.ring_index),
                    )
                else:
                    view_infos[i] = self.undistort_pool.apply_async(
                        lambda x: x, (view_infos[i],)
                    )
        if not self.gpu_frames:
            self.ring_index = (self.ring_index + 1) % self.ringbuffer_size

        self.q_messages_pre.put(
            messages.UpdateScene(
                view_infos=view_infos, viz_camera=viz_camera, viz_imshape=viz_imshape
            ),
            block=True,  # Always block here since we checked above
        )

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
        if not self.gpu_frames:
            self.undistort_pool.finish()
        self.q_messages_pre.put(messages.Quit())
        self._join()
        self.posedata_waiter_thread.join()
        self.visualizer_process.join()

    def _join(self):
        if not self.gpu_frames:
            self.undistort_pool.join()
        self.q_messages_pre.join()
        self.q_messages_post.join()

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager protocol."""
        self.close()


class _ImmediateResult:
    """Wraps a pre-computed value to match AsyncResult.get() interface."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _main_visualize_gl(*args, **kwargs):
    _terminate_on_parent_death()
    from poseviz.gl.renderer import PoseVizGLSide

    viz = PoseVizGLSide(*args, **kwargs)
    viz.run_loop()


_main_visualize = _main_visualize_gl


def _terminate_on_parent_death():
    # Make sure the visualizer process is terminated when the parent process dies, for whatever reason
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)

    # ignore sigterm and sigint
    # without this, a single CTRL+C leaves the process hanging
    # signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def main_posedata_waiter(q_posedata_pre, q_posedata_post):
    import time
    _t_get = 0.0
    _t_resolve = 0.0
    _t_put = 0.0
    _t_count = 0
    _t_last = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        msg = q_posedata_pre.get()
        t1 = time.perf_counter()
        if isinstance(msg, messages.UpdateScene):
            msg.view_infos = [v.get() for v in msg.view_infos]
        t2 = time.perf_counter()
        q_posedata_post.put(msg)
        t3 = time.perf_counter()
        q_posedata_pre.task_done()

        if isinstance(msg, messages.UpdateScene):
            _t_get += t1 - t0
            _t_resolve += t2 - t1
            _t_put += t3 - t2
            _t_count += 1
            if t3 - _t_last >= 0.5 and _t_count > 0:
                n = _t_count
                logger.debug(
                    f"[waiter] get={_t_get/n*1000:.1f}ms "
                    f"resolve={_t_resolve/n*1000:.1f}ms "
                    f"put={_t_put/n*1000:.1f}ms "
                    f"total={(_t_get+_t_resolve+_t_put)/n*1000:.1f}ms "
                    f"n={n} ({n/(t3-_t_last):.0f}fps)"
                )
                _t_get = _t_resolve = _t_put = 0.0
                _t_count = 0
                _t_last = t3

        if isinstance(msg, messages.Quit):
            return
