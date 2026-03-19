import itertools
import logging
import queue

import glfw
import numpy as np
import simplepyutils as spu
import framepump

from poseviz import messages
from poseviz.gl.window import create_window
from poseviz.gl.transforms import set_world_up, camera_to_gl_mvp, camera_to_gl_view
from poseviz.gl.renderables.base import ShaderRenderable
from poseviz.gl.view_visualizer import ViewVisualizer
from poseviz.gl.renderables import GroundPlaneRenderable
from poseviz.gl.pyramid_picker import PyramidPicker
from poseviz.gl.viewport import Viewport
from poseviz.gl.terrain_camera import TerrainCamera
import poseviz.colors
import poseviz.draw2d

logger = logging.getLogger(__name__)


class PoseVizGLSide:
    """OpenGL renderer that runs in the visualizer process."""

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
        render_resolution,
        use_virtual_display,
        show_virtual_display,
        show_ground_plane,
        main_cam_value,
        camera_view_padding,
        body_model_faces,
        show_image,
        image_plane_distance,
        frame_rings,
        flying_mode="camera",
        headless=False,
        gpu_encode=True,
        fullscreen=False,
    ):
        self.q_messages = q_messages
        self.video_writer = None  # Will be set based on mode
        self.headless = headless
        self.gpu_encode = gpu_encode
        self.fullscreen = fullscreen
        self.windowed_pos = None  # Stored position when entering fullscreen
        self.windowed_size = None  # Stored size when entering fullscreen

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
        self.resolution = resolution  # Window/display resolution
        self.render_resolution = (
            render_resolution  # FBO/video resolution (can be larger)
        )
        self.use_virtual_display = use_virtual_display
        self.show_virtual_display = show_virtual_display
        self.main_cam_value = main_cam_value
        self.camera_view_padding = camera_view_padding
        self.body_model_faces = body_model_faces
        self.i_pred_frame = 0
        self.camera_trajectory = []
        self.camera_trajectory_path = None
        self.show_image = show_image
        self.image_plane_distance = image_plane_distance
        self.frame_rings = frame_rings

        # Will be initialized in run_loop
        self.window = None
        self.ctx = None
        self.fbo = None  # Offscreen framebuffer (resolved, for video output)
        self.fbo_msaa = None  # Multisampled framebuffer for antialiased rendering
        self.samples = 4  # MSAA sample count
        self.blit_prog = None  # Shader for blitting FBO to screen with scaling
        self.blit_vao = None
        self.view_visualizers = None
        self.ground_renderer = None

        # Current view camera (deltacamera.Camera) and image shape
        self.current_camera = None
        self.current_imshape = None

        # Terrain camera for free-fly navigation
        self.terrain_camera = TerrainCamera(flying_mode=flying_mode)

        # Mouse interaction state
        self.mouse_start_pos = None
        self.mouse_button_pressed = None
        self.mouse_mods = 0
        self.mouse_start_viewport = None

        # Whether a scripted viz_camera is driving the view
        self._has_viz_camera = False
        # Whether the user is currently holding a mouse button for camera control
        self._user_dragging = False

        # Split-screen mode: left = original camera, right = free camera
        self.split_screen = False

        # Camera picking state
        self.pyramid_picker = None
        self.last_click_time = 0.0
        self.last_click_cam = -1
        self.selected_camera = -1  # Currently selected camera index
        self.double_click_threshold = 0.3  # seconds

        # Viewports
        self.viewports = []

        # Flag to push terrain camera state after next initialization
        self._push_after_terrain_init = False

    def run_loop(self):
        set_world_up(self.world_up)

        if self.headless:
            self._run_loop_headless()
        else:
            self._run_loop_windowed()

    def _run_loop_windowed(self):
        """Main loop with GLFW window for interactive use."""
        self.window, self.ctx = create_window(
            self.resolution[0],
            self.resolution[1],
            "PoseViz",
            fullscreen=self.fullscreen,
        )

        # Update resolution from actual window size (fullscreen uses monitor resolution)
        self.resolution = glfw.get_framebuffer_size(self.window)

        # Resolve render_resolution: None means match display resolution
        if self.render_resolution is None:
            self.render_resolution = self.resolution

        # FBO at render_resolution (can be larger than window for high-res video output)
        render_w, render_h = self.render_resolution

        # Create multisampled FBO for antialiased rendering
        self.fbo_msaa = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.renderbuffer((render_w, render_h), 4, samples=self.samples)
            ],
            depth_attachment=self.ctx.depth_renderbuffer(
                (render_w, render_h), samples=self.samples
            ),
        )

        # Create resolved FBO with texture (for video encoding and display)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((render_w, render_h), 4)],  # RGBA
            depth_attachment=self.ctx.depth_texture((render_w, render_h)),
        )

        # Create blit shader for scaling FBO to screen
        self._init_blit_shader()

        # Use GLVideoWriter for zero-copy encoding if gpu_encode, else framepump
        if self.gpu_encode:
            from framepump.video_writing_gl import GLVideoWriter

            self.video_writer = GLVideoWriter()
        else:
            self.video_writer = framepump.VideoWriter()

        # Set up callbacks
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

        self._init_renderers()

        # Main loop
        frame_time = 1.0 / self.fps
        last_time = glfw.get_time()
        should_quit = False

        # FPS measurement
        fps_update_interval = 0.5  # Update FPS display every 0.5 seconds
        fps_last_update = glfw.get_time()
        fps_frame_count = 0
        content_frame_count = 0

        # Timing instrumentation
        import time
        _t_handle = 0.0
        _t_render = 0.0
        _t_swap = 0.0
        _t_content_frames = 0
        _t_empty_frames = 0
        _t_log_interval = 0.5
        _t_last_log = time.perf_counter()

        pending_msg = None
        while not glfw.window_should_close(self.window) and not should_quit:
            glfw.poll_events()

            now = glfw.get_time()
            remaining = frame_time - (now - last_time)

            if self.paused:
                if remaining > 0.001:
                    time.sleep(min(remaining, 0.002))
                    continue
                msg = messages.Nothing()
            elif remaining > 0.001:
                # Buffer content during idle time, but don't render yet
                if pending_msg is None:
                    try:
                        pending_msg = self.q_messages.get(timeout=min(remaining, 0.002))
                    except queue.Empty:
                        pass
                else:
                    time.sleep(min(remaining, 0.002))
                continue
            else:
                # Frame time elapsed — use buffered msg or wait briefly for late content
                if pending_msg is not None:
                    msg = pending_msg
                    pending_msg = None
                else:
                    try:
                        msg = self.q_messages.get(timeout=0.004)
                    except queue.Empty:
                        msg = messages.Nothing()

            last_time = glfw.get_time()

            # Process messages
            t0 = time.perf_counter()
            has_content = isinstance(msg, messages.UpdateScene)
            should_quit = self.handle_message(msg)
            t1 = time.perf_counter()

            # Render
            self.render()
            t2 = time.perf_counter()
            glfw.swap_buffers(self.window)
            t3 = time.perf_counter()
            glfw.poll_events()

            _t_handle += t1 - t0
            _t_render += t2 - t1
            _t_swap += t3 - t2
            if has_content:
                _t_content_frames += 1
            else:
                _t_empty_frames += 1

            # FPS measurement
            fps_frame_count += 1
            if has_content:
                content_frame_count += 1
            if now - fps_last_update >= fps_update_interval:
                dt = now - fps_last_update
                render_fps = fps_frame_count / dt
                content_fps = content_frame_count / dt
                glfw.set_window_title(
                    self.window,
                    f"PoseViz - render {render_fps:.0f} fps | content {content_fps:.0f} fps",
                )
                fps_frame_count = 0
                content_frame_count = 0
                fps_last_update = now

            # Timing log
            t_now = time.perf_counter()
            if t_now - _t_last_log >= _t_log_interval:
                n = _t_content_frames + _t_empty_frames
                if n > 0:
                    logger.debug(
                        f"[renderer] handle={_t_handle/n*1000:.1f}ms "
                        f"render={_t_render/n*1000:.1f}ms "
                        f"swap={_t_swap/n*1000:.1f}ms "
                        f"content={_t_content_frames} empty={_t_empty_frames}"
                    )
                _t_handle = _t_render = _t_swap = 0.0
                _t_content_frames = _t_empty_frames = 0
                _t_last_log = t_now

        glfw.terminate()

    def _run_loop_headless(self):
        """Main loop for headless rendering (no window, GPU accelerated).

        Uses GLFW with invisible window to get a proper GL context that can be
        used with NVENC for zero-copy GPU encoding via GLVideoWriter.
        """
        import moderngl

        width, height = self.resolution

        # Initialize GLFW with invisible window (needed for GL context)
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self.window = glfw.create_window(width, height, "PoseViz Headless", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # Create moderngl context from the GLFW window
        self.ctx = moderngl.create_context()

        # Resolve render_resolution: None means match display resolution
        if self.render_resolution is None:
            self.render_resolution = self.resolution
        render_w, render_h = self.render_resolution

        # Create multisampled FBO for antialiased rendering
        self.fbo_msaa = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.renderbuffer((render_w, render_h), 4, samples=self.samples)
            ],
            depth_attachment=self.ctx.depth_renderbuffer(
                (render_w, render_h), samples=self.samples
            ),
        )

        # Create resolved FBO with texture (for zero-copy encoding)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((render_w, render_h), 4)],  # RGBA
            depth_attachment=self.ctx.depth_texture((render_w, render_h)),
        )
        self.fbo_msaa.use()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Initialize blit shader for flipped encoding
        self._init_blit_shader()

        self._init_renderers()

        # Use GLVideoWriter for zero-copy encoding if gpu_encode, else framepump
        if self.gpu_encode:
            from framepump.video_writing_gl import GLVideoWriter

            self.video_writer = GLVideoWriter()
        else:
            self.video_writer = framepump.VideoWriter()

        # Simple loop - block-wait for messages, no spinning
        should_quit = False
        while not should_quit:
            # Block until message arrives (no fps throttling, max throughput)
            msg = self.q_messages.get()
            should_quit = self.handle_message(msg)
            self.render()

        # Cleanup
        glfw.terminate()

    def _init_renderers(self):
        """Initialize view visualizers and other renderers."""
        # Initialize view visualizers
        self.view_visualizers = [
            ViewVisualizer(
                self.ctx,
                self.joint_info,
                self.camera_type,
                show_image=self.show_image,
                high_quality=self.high_quality,
                show_field_of_view=self.show_field_of_view,
                show_camera_wireframe=self.show_camera_wireframe,
                body_model_faces=self.body_model_faces,
                image_plane_distance=self.image_plane_distance,
            )
            for _ in range(self.n_views)
        ]

        # Ground plane
        if self.show_ground_plane:
            self.ground_renderer = GroundPlaneRenderable(
                self.ctx, self.ground_plane_height
            )

        # Camera picker for click selection
        self.pyramid_picker = PyramidPicker(self.ctx, self.resolution)

        # Initialize viewports
        self._rebuild_viewports()

    def _init_blit_shader(self):
        """Initialize shader for blitting FBO to screen with scaling."""
        self.blit_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 uv;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 uv;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(tex, uv);
                }
            """,
        )
        # Fullscreen quad (two triangles) - V flipped to compensate for projection Y flip
        import array

        vertices = array.array(
            "f",
            [
                # pos        u    v (flipped: bottom=1, top=0)
                -1,
                -1,
                0,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                0,
                -1,
                -1,
                0,
                1,
                1,
                1,
                1,
                0,
                -1,
                1,
                0,
                0,
            ],
        )
        vbo = self.ctx.buffer(vertices.tobytes())
        self.blit_vao = self.ctx.vertex_array(
            self.blit_prog, [(vbo, "2f 2f", "in_pos", "in_uv")]
        )

    def handle_message(self, msg) -> bool:
        """Handle a message. Returns True if should quit."""
        if isinstance(msg, messages.Nothing):
            if self.current_viewinfos is not None:
                self.update_view_camera()
            if self.paused:
                self.capture_frame()
            return False

        elif isinstance(msg, messages.UpdateScene):
            self.update_visu(msg.view_infos)
            self.update_view_camera(msg.viz_camera, msg.viz_imshape)
            self.capture_frame()

            if self.step_one_by_one:
                self.step_one_by_one = False
                self.paused = True

            self.i_pred_frame += 1
            self.q_messages.task_done()
            return False

        elif isinstance(msg, messages.ReinitCameraView):
            if self.snap_to_cam_on_scene_change:
                self.initialized_camera = False
                self.current_viewinfos = None
            self.q_messages.task_done()
            return False

        elif isinstance(msg, messages.StartSequence):
            if msg.video_path is not None:
                # GLVideoWriter doesn't have use_gpu param, regular VideoWriter does
                from framepump.video_writing_gl import GLVideoWriter

                is_gl = isinstance(self.video_writer, GLVideoWriter)
                self.video_writer.start_sequence(
                    msg.video_path,
                    fps=msg.fps,
                    audio_source_path=msg.audio_source_path,
                    **({} if is_gl else dict(gpu=True)),
                )
            self.camera_trajectory_path = msg.camera_trajectory_path
            self.i_pred_frame = 0
            self.q_messages.task_done()
            return False

        elif isinstance(msg, messages.Pause):
            self.paused = True
            self.q_messages.task_done()
            return False

        elif isinstance(msg, messages.Resume):
            self.paused = False
            self.q_messages.task_done()
            return False

        elif isinstance(msg, messages.EndSequence):
            if self.camera_trajectory_path is not None:
                spu.dump_pickle(self.camera_trajectory, self.camera_trajectory_path)
            self.video_writer.end_sequence()
            self.q_messages.task_done()
            return False

        elif isinstance(msg, messages.Quit):
            if self.camera_trajectory_path is not None:
                spu.dump_pickle(self.camera_trajectory, self.camera_trajectory_path)
            self.video_writer.close()
            self.current_viewinfos = None
            # Drain any remaining messages to release CUDA IPC tensor references
            while True:
                try:
                    leftover = self.q_messages.get_nowait()
                    self.q_messages.task_done()
                except queue.Empty:
                    break
            self.q_messages.task_done()
            return True

        else:
            raise ValueError("Unknown message:", msg)

    def receive_message(self):
        """Get message from queue. Returns Nothing if paused or queue empty."""
        if self.paused:
            return messages.Nothing()

        try:
            return self.q_messages.get_nowait()
        except queue.Empty:
            return messages.Nothing()

    def update_visu(self, view_infos):
        import time
        _tv0 = time.perf_counter()

        self.update_num_views(len(view_infos))

        # Reconstruct frames from shared memory (CPU path) or keep as CUDA tensor (GPU path)
        for i, v in enumerate(view_infos):
            if v.frame is not None and isinstance(v.frame, tuple):
                shape, dtype, index = v.frame
                v.frame = self.frame_rings[i].read(index, shape)

        # Gather poses from displayed views
        pose_displayed_view_infos = (
            view_infos
            if self.pose_displayed_cam_id is None
            else [view_infos[self.pose_displayed_cam_id]]
        )

        all_poses = [p for v in pose_displayed_view_infos for p in v.poses]
        all_poses_true = [p for v in pose_displayed_view_infos for p in v.poses_true]
        all_poses_alt = [p for v in pose_displayed_view_infos for p in v.poses_alt]
        all_vertices = [p for v in pose_displayed_view_infos for p in v.vertices]
        all_vertices_true = [
            p for v in pose_displayed_view_infos for p in v.vertices_true
        ]
        all_vertices_alt = [
            p for v in pose_displayed_view_infos for p in v.vertices_alt
        ]

        self.current_viewinfos = view_infos

        # Draw 2D overlays on frames
        if self.multicolor_detections:
            box_colors = poseviz.colors.cycle_over_colors(False)
        else:
            box_colors = itertools.repeat((31, 119, 180))

        joint_names, joint_edges = self.joint_info

        _tv1 = time.perf_counter()

        for i_viz, (view_info, viz) in enumerate(
            zip(view_infos, self.view_visualizers)
        ):
            poses = all_poses if viz is self.view_visualizers[0] else None
            poses_true = all_poses_true if viz is self.view_visualizers[0] else None
            poses_alt = all_poses_alt if viz is self.view_visualizers[0] else None
            vertices = all_vertices if viz is self.view_visualizers[0] else None
            vertices_true = (
                all_vertices_true if viz is self.view_visualizers[0] else None
            )
            vertices_alt = all_vertices_alt if viz is self.view_visualizers[0] else None

            # 2D overlays only work on numpy frames (CPU path)
            is_numpy_frame = isinstance(view_info.frame, np.ndarray)

            max_size = max(view_info.frame.shape[:2])
            if max_size < 512:
                thickness = 1
            elif max_size < 1024:
                thickness = 2
            else:
                thickness = 3

            if is_numpy_frame and self.draw_detections:
                for color, box in zip(box_colors, view_info.boxes):
                    poseviz.draw2d.draw_box(
                        view_info.frame, box, color, thickness=thickness
                    )

            if is_numpy_frame and self.draw_2d_pose:
                pose_groups = [
                    view_info.poses,
                    view_info.poses_true,
                    view_info.poses_alt,
                ]
                colors = [
                    poseviz.colors.green,
                    poseviz.colors.red,
                    poseviz.colors.orange,
                ]
                for pose_group, color in zip(pose_groups, colors):
                    for pose in pose_group:
                        pose2d = view_info.camera.world_to_image(pose)
                        poseviz.draw2d.draw_stick_figure_2d_inplace(
                            view_info.frame, pose2d, joint_edges, thickness, color=color
                        )

            _tv2 = time.perf_counter()

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

            _tv3 = time.perf_counter()

            # Update picker pyramid for this camera
            if self.pyramid_picker is not None:
                self.pyramid_picker.update_camera(
                    i_viz,
                    view_info.camera,
                    view_info.frame.shape,
                    self.image_plane_distance,
                )

            # Release frame data, keep only shape (for update_view_camera).
            # Essential for CUDA IPC tensors: tells PyTorch the receiver is done.
            view_info.frame_shape = view_info.frame.shape
            view_info.frame = None

        _tv4 = time.perf_counter()
        logger.debug(
            f"  [update_visu] gather={(_tv1-_tv0)*1000:.1f}ms "
            f"draw2d={(_tv2-_tv1)*1000:.1f}ms "
            f"viz.update={(_tv3-_tv2)*1000:.1f}ms "
            f"picker={(_tv4-_tv3)*1000:.1f}ms "
            f"total={(_tv4-_tv0)*1000:.1f}ms"
        )

    def update_num_views(self, new_n_views):
        if new_n_views > self.n_views:
            self.view_visualizers += [
                ViewVisualizer(
                    self.ctx,
                    self.joint_info,
                    self.camera_type,
                    show_image=self.show_image,
                    high_quality=self.high_quality,
                    show_field_of_view=self.show_field_of_view,
                    show_camera_wireframe=self.show_camera_wireframe,
                    body_model_faces=self.body_model_faces,
                    image_plane_distance=self.image_plane_distance,
                )
                for _ in range(new_n_views - self.n_views)
            ]
            self.n_views = new_n_views
        elif new_n_views < self.n_views:
            for viz in self.view_visualizers[new_n_views:]:
                viz.destroy()
            del self.view_visualizers[new_n_views:]
            self.n_views = new_n_views

    def update_view_camera(self, camera=None, imshape=None):
        main_view_info = self.current_viewinfos[self.main_cam]

        if camera is not None:
            if imshape is None:
                imshape = main_view_info.frame_shape
            self.current_camera = camera
            self.current_imshape = imshape
            # Track that we have a scripted viz_camera driving the view.
            # Keep terrain camera in sync so it's at the right position when
            # the user starts dragging, but don't overwrite during an active drag.
            self._has_viz_camera = True
            if self.camera_type == "free" and not self._user_dragging:
                self._init_terrain_camera(camera)
        elif self.camera_type == "original" or (
            self.camera_type == "free" and not self.initialized_camera
        ):
            self.current_camera = main_view_info.camera
            self.current_imshape = main_view_info.frame_shape

            # Initialize interactive camera state from first frame's camera
            if self.camera_type == "free" and not self.initialized_camera:
                self._init_terrain_camera(main_view_info.camera)
                # Set snapped position as pending checkpoint if flagged
                if self._push_after_terrain_init:
                    self.terrain_camera.set_pending_checkpoint()
                    self._push_after_terrain_init = False

        elif self.camera_type == "bird" and not self.initialized_camera:
            # Bird's eye view (experimental)
            cam = main_view_info.camera
            pivot = cam.t + cam.R[2] * 2000
            camera_view = (
                cam.copy(optical_center=(cam.t - pivot) * 1.35 + pivot)
                .orbited_around(pivot, np.deg2rad(-25), "vertical")
                .orbited_around(pivot, np.deg2rad(-15), "horizontal")
            )
            self.current_camera = camera_view
            self.current_imshape = main_view_info.frame_shape

        self.initialized_camera = True

        # If terrain camera is snapped, update it to follow the display camera
        if self.terrain_camera.is_snapped:
            snapped_idx = self.terrain_camera.snapped_to
            if snapped_idx < len(self.current_viewinfos):
                self._init_terrain_camera(self.current_viewinfos[snapped_idx].camera)

    def _init_terrain_camera(self, cam=None):
        """Initialize terrain camera state from a reference camera."""
        if cam is None:
            if self.current_viewinfos is None:
                return
            cam = self.current_viewinfos[self.main_cam].camera

        self.terrain_camera.init_from_camera(cam)

    def _rebuild_viewports(self):
        """Rebuild viewport list based on current mode and resolution."""
        width, height = self.resolution
        self.viewports = []

        if self.split_screen:
            half_width = width // 2

            # Left viewport: original camera (non-interactive)
            self.viewports.append(
                Viewport(
                    name="original",
                    bounds=(0, 0, half_width, height),
                    get_view_proj=lambda: self._get_original_view_proj(
                        (height, half_width)
                    ),
                    get_matrices=lambda: (
                        self._get_original_view_proj((height, half_width)),
                        self._get_original_view(),
                    ),
                    interactive=False,
                )
            )

            # Right viewport: terrain camera (interactive)
            self.viewports.append(
                Viewport(
                    name="terrain",
                    bounds=(half_width, 0, width - half_width, height),
                    get_view_proj=lambda: self._get_terrain_view_proj(
                        (height, width - half_width)
                    ),
                    get_matrices=lambda: (
                        self._get_terrain_view_proj((height, width - half_width)),
                        self._get_terrain_view((height, width - half_width)),
                    ),
                    interactive=True,
                )
            )
        else:
            # Single viewport covering entire screen
            is_free = self.camera_type == "free"
            imshape = (height, width)
            self.viewports.append(
                Viewport(
                    name="terrain" if is_free else "original",
                    bounds=(0, 0, width, height),
                    get_view_proj=lambda: (
                        self._get_terrain_view_proj(imshape)
                        if self._use_terrain_camera()
                        else self._get_original_view_proj(imshape)
                    ),
                    get_matrices=lambda: (
                        (self._get_terrain_view_proj(imshape), self._get_terrain_view(imshape))
                        if self._use_terrain_camera()
                        else (self._get_original_view_proj(imshape), self._get_original_view())
                    ),
                    interactive=is_free,
                )
            )

    def _use_terrain_camera(self):
        """Whether to render from the terrain camera (vs original/viz camera)."""
        return (
            self.camera_type == "free"
            and self.terrain_camera.initialized
            and (self._user_dragging or not self._has_viz_camera)
        )

    def _get_original_view_proj(self, imshape_override=None):
        """Get view-projection matrix for the original (display) camera."""
        if self.current_camera is None:
            return np.eye(4, dtype=np.float32)
        imshape = imshape_override if imshape_override else self.current_imshape
        if imshape is None:
            imshape = self.resolution[::-1]  # (height, width)
        return camera_to_gl_mvp(self.current_camera, imshape)

    def _get_original_view(self):
        """Get view matrix for the original (display) camera."""
        if self.current_camera is None:
            return np.eye(4, dtype=np.float32)
        return camera_to_gl_view(self.current_camera)

    def _get_viewport_at(self, x: int, y: int):
        """Find which viewport contains the given screen coordinates."""
        for viewport in self.viewports:
            if viewport.contains(x, y):
                return viewport
        return None

    def _get_interactive_viewport(self):
        """Get the interactive viewport (for mouse controls)."""
        for viewport in self.viewports:
            if viewport.interactive:
                return viewport
        return None

    def render(self):
        # Render to multisampled FBO at render_resolution
        if self.fbo_msaa is not None:
            self.fbo_msaa.use()
            render_w, render_h = self.render_resolution
            self.ctx.viewport = (0, 0, render_w, render_h)

        self.ctx.clear(1.0, 1.0, 1.0)  # White background

        if self.current_camera is None:
            return

        # Render each viewport (supports single view and split-screen)
        use_fbo = self.fbo_msaa is not None
        if use_fbo:
            flip_y = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
            scale_x = render_w / self.resolution[0]
            scale_y = render_h / self.resolution[1]

        for viewport in self.viewports:
            if use_fbo:
                # Scale viewport bounds from display coords to FBO coords
                self.ctx.viewport = (
                    int(viewport.x * scale_x), int(viewport.y * scale_y),
                    int(viewport.width * scale_x), int(viewport.height * scale_y),
                )
                view_proj, view = viewport.get_matrices()
                view_proj = view_proj @ flip_y
            else:
                self.ctx.viewport = viewport.bounds
                view_proj, view = viewport.get_matrices()

            ShaderRenderable._current_view = view.astype(np.float32).tobytes()
            self._render_scene(view_proj)

        # Resolve multisampled FBO to regular FBO (for video encoding)
        if self.fbo_msaa is not None and self.fbo is not None:
            self.ctx.copy_framebuffer(self.fbo, self.fbo_msaa)

        # In windowed mode, blit FBO to screen for display (with scaling)
        if self.fbo is not None and not self.headless:
            self.ctx.screen.use()
            self.ctx.viewport = (0, 0, *self.resolution)
            self.ctx.disable(self.ctx.DEPTH_TEST)  # Disable depth for 2D blit
            # Bind FBO texture and render fullscreen quad with blit shader
            self.fbo.color_attachments[0].use(location=0)
            self.blit_prog["tex"].value = 0
            self.blit_vao.render()
            self.ctx.enable(self.ctx.DEPTH_TEST)  # Re-enable for next frame

    def _render_scene(self, view_proj):
        """Render the full scene with a given view-projection matrix."""
        if self.ground_renderer:
            self.ground_renderer.render(view_proj)

        for viz in self.view_visualizers:
            viz.render(view_proj)

    def capture_frame(self):
        if not self.video_writer.accepts_new_frames:
            return

        # Check if using GLVideoWriter (zero-copy) or regular VideoWriter
        from framepump.video_writing_gl import GLVideoWriter

        if isinstance(self.video_writer, GLVideoWriter) and self.fbo is not None:
            # Zero-copy: encode directly from FBO
            # Note: FBO is upside-down for video (OpenGL Y=0 at bottom)
            # GLVideoWriter handles the flip in the CUDA/NVENC path
            self.ctx.finish()
            self.video_writer.append_data(self.fbo.color_attachments[0])
        else:
            # CPU fallback: read pixels and flip
            width, height = self.render_resolution if self.fbo else self.resolution
            if self.fbo is not None:
                data = self.fbo.read(components=3)
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                frame = frame[::-1]  # Flip for correct video orientation
            else:
                data = self.ctx.screen.read(components=3)
                frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                frame = frame[::-1]  # Screen also needs flip
            self.video_writer.append_data(frame)

    # --- Input callbacks ---

    def _on_key(self, window, key, _scancode, action, _mods):
        # Keys that support repeat for continuous adjustment when held
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:  # + key
                self.terrain_camera.adjust_fov(-5.0)  # Decrease FOV = zoom in
                return
            elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:  # - key
                self.terrain_camera.adjust_fov(5.0)  # Increase FOV = zoom out
                return
            elif key in (
                glfw.KEY_UP,
                glfw.KEY_DOWN,
                glfw.KEY_LEFT,
                glfw.KEY_RIGHT,
                glfw.KEY_PAGE_UP,
                glfw.KEY_PAGE_DOWN,
            ):
                self._handle_flying_key(key)
                return

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)
        elif key == glfw.KEY_F11:
            self._toggle_fullscreen()
        elif key == glfw.KEY_X:
            self.paused = not self.paused
        elif key == glfw.KEY_T:
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = None
        elif key == glfw.KEY_C:
            self.step_one_by_one = True
            self.paused = False
        elif key == glfw.KEY_N:
            self.main_cam = (self.main_cam + 1) % self.n_views
            self.main_cam_value.value = self.main_cam
            self.initialized_camera = False
        elif key == glfw.KEY_M:
            if self.pose_displayed_cam_id is None:
                self.pose_displayed_cam_id = self.main_cam
            else:
                self.pose_displayed_cam_id = (
                    self.pose_displayed_cam_id + 1
                ) % self.n_views
        elif key == glfw.KEY_O:
            self.main_cam = (self.main_cam + 1) % self.n_views
            self.main_cam_value.value = self.main_cam
            self.pose_displayed_cam_id = self.main_cam
            self.initialized_camera = False
        elif key == glfw.KEY_Z:
            self.camera_type = "original" if self.camera_type != "original" else "free"
            self.initialized_camera = False
        elif key == glfw.KEY_U:
            # Show just main cam pred
            self.pose_displayed_cam_id = self.main_cam
        elif key == glfw.KEY_D or key == glfw.KEY_G:
            # Snap to nearest camera
            if self.current_viewinfos is not None:
                display_cameras = [vi.camera for vi in self.current_viewinfos]
                viewing_cam_pos = self.terrain_camera.get_position()
                new_cam = int(
                    np.argmin(
                        [np.linalg.norm(viewing_cam_pos - c.t) for c in display_cameras]
                    )
                )
                if new_cam != self.main_cam or not self.terrain_camera.is_snapped:
                    self.terrain_camera.push_state()  # Save where we were
                    self.terrain_camera.snap_to(new_cam)  # Snap to follow this camera
                    self._push_after_terrain_init = (
                        True  # Set pending checkpoint after init
                    )
                    self.main_cam = new_cam
                    self.main_cam_value.value = self.main_cam
                    self.initialized_camera = False
                if key == glfw.KEY_G:
                    # Also display predicted pose from this nearest camera
                    self.pose_displayed_cam_id = self.main_cam
        elif key == glfw.KEY_TAB:
            self.split_screen = not self.split_screen
            # Initialize terrain camera if needed
            if self.split_screen and not self.terrain_camera.initialized:
                self._init_terrain_camera()
            self._rebuild_viewports()
        elif key >= glfw.KEY_1 and key <= glfw.KEY_9:
            cam_idx = key - glfw.KEY_1
            if cam_idx < self.n_views and (
                cam_idx != self.main_cam or not self.terrain_camera.is_snapped
            ):
                self.terrain_camera.push_state()  # Save where we were
                self.terrain_camera.snap_to(cam_idx)  # Snap to follow this camera
                self._push_after_terrain_init = (
                    True  # Set pending checkpoint after init
                )
                self.main_cam = cam_idx
                self.main_cam_value.value = self.main_cam
                self.initialized_camera = False

    def _on_mouse_button(self, window, button, action, mods):
        if action == glfw.PRESS:
            # Handle mouse back/forward buttons for history navigation
            if button == glfw.MOUSE_BUTTON_4:  # Back button
                self.terrain_camera.go_back()
                return
            elif button == glfw.MOUSE_BUTTON_5:  # Forward button
                self.terrain_camera.go_forward()
                return

            self.mouse_button_pressed = button
            self.mouse_mods = mods
            self._user_dragging = True
            x, y = glfw.get_cursor_pos(window)
            self.mouse_start_pos = (x, y)
            self.terrain_camera.begin_drag()
            # Track which viewport was clicked for mouse controls
            self.mouse_start_viewport = self._get_viewport_at(int(x), int(y))
        elif action == glfw.RELEASE:
            # Check if this was a click (not a drag)
            if button == glfw.MOUSE_BUTTON_LEFT and self.mouse_start_pos is not None:
                x, y = glfw.get_cursor_pos(window)
                dx = abs(x - self.mouse_start_pos[0])
                dy = abs(y - self.mouse_start_pos[1])

                # If mouse barely moved, treat as click for picking
                if dx < 5 and dy < 5:
                    self._handle_camera_pick(int(x), int(y))

            self.mouse_button_pressed = None
            self.mouse_start_pos = None
            self._user_dragging = False
            self._has_viz_camera = False

    def _on_mouse_move(self, window, x, y):
        if self.mouse_start_pos is None:
            return
        # Only apply camera controls if drag started in interactive viewport
        if (
            self.mouse_start_viewport is None
            or not self.mouse_start_viewport.interactive
        ):
            return

        dx = x - self.mouse_start_pos[0]
        dy = y - self.mouse_start_pos[1]

        if self.mouse_button_pressed == glfw.MOUSE_BUTTON_LEFT and self.mouse_mods & glfw.MOD_SHIFT:
            self.terrain_camera.look_around(dx, dy)
        elif self.mouse_button_pressed == glfw.MOUSE_BUTTON_LEFT:
            self.terrain_camera.orbit(dx, dy)
        elif self.mouse_button_pressed == glfw.MOUSE_BUTTON_MIDDLE:
            self.terrain_camera.pan(dx, dy)
        elif self.mouse_button_pressed == glfw.MOUSE_BUTTON_RIGHT:
            self.terrain_camera.zoom_drag(dy)

    def _on_scroll(self, window, _xoffset, yoffset):
        # Only apply scroll zoom if cursor is over interactive viewport
        x, y = glfw.get_cursor_pos(window)
        viewport = self._get_viewport_at(int(x), int(y))
        if viewport is None or not viewport.interactive:
            return
        self.terrain_camera.zoom_scroll(yoffset)

    def _on_resize(self, window, width, height):
        """Handle window resize."""
        if width > 0 and height > 0:
            old_resolution = self.resolution
            self.resolution = (width, height)
            self.ctx.viewport = (0, 0, width, height)

            # If render_resolution was matching display, keep matching
            if self.render_resolution == old_resolution:
                self.render_resolution = self.resolution
                self._recreate_fbos()

            # Resize picker framebuffer
            if self.pyramid_picker is not None:
                self.pyramid_picker.resize(width, height)
            # Rebuild viewports for new resolution
            self._rebuild_viewports()

    def _handle_camera_pick(self, x: int, y: int):
        """Handle camera picking at screen position (x, y)."""
        if self.pyramid_picker is None or self.current_camera is None:
            return

        # Find which viewport contains the click
        viewport = self._get_viewport_at(x, y)
        if viewport is None:
            return

        # Transform to viewport-local coordinates for picker
        local_x, local_y = viewport.to_local(x, y)

        # Resize picker to viewport size (picker framebuffer must match viewport)
        self.pyramid_picker.resize(viewport.width, viewport.height)

        # Get view-proj from the clicked viewport
        view_proj = viewport.get_view_proj()

        # Do the pick with local coordinates
        cam_idx = self.pyramid_picker.pick(local_x, local_y, view_proj)

        if cam_idx >= 0:
            current_time = glfw.get_time()

            # Check for double-click
            if (
                cam_idx == self.last_click_cam
                and current_time - self.last_click_time < self.double_click_threshold
            ):
                # Double-click: jump to this camera
                self._jump_to_camera(cam_idx)
                self.last_click_cam = -1
                self.last_click_time = 0.0
            else:
                # Single click: select this camera
                self.selected_camera = cam_idx
                self.last_click_cam = cam_idx
                self.last_click_time = current_time
        else:
            # Clicked on nothing - deselect
            self.selected_camera = -1

    def _jump_to_camera(self, cam_idx: int):
        """Jump the view to a specific camera (snaps to follow it)."""
        if self.current_viewinfos is None or cam_idx >= len(self.current_viewinfos):
            return

        # Save current position to history before jumping
        self.terrain_camera.push_state()

        # Set main camera to this index
        self.main_cam = cam_idx
        self.main_cam_value.value = cam_idx
        self.initialized_camera = False

        # Snap terrain camera to follow this display camera
        self.terrain_camera.snap_to(cam_idx)
        cam = self.current_viewinfos[cam_idx].camera
        self._init_terrain_camera(cam)

        # Set snapped state as pending checkpoint (committed on first user movement)
        self.terrain_camera.set_pending_checkpoint()

    def _handle_flying_key(self, key):
        """Handle arrow + page up/down flying controls."""
        if self.camera_type != "free" and not self.split_screen:
            return

        key_to_direction = {
            glfw.KEY_UP: "forward",
            glfw.KEY_DOWN: "backward",
            glfw.KEY_LEFT: "left",
            glfw.KEY_RIGHT: "right",
            glfw.KEY_PAGE_UP: "up",
            glfw.KEY_PAGE_DOWN: "down",
        }
        if key in key_to_direction:
            self.terrain_camera.fly(key_to_direction[key])

    def _get_terrain_view_proj(self, imshape=None):
        """Get terrain camera view-projection matrix."""
        if imshape is None:
            imshape = (self.resolution[1], self.resolution[0])
        return self.terrain_camera.get_view_proj(imshape)

    def _get_terrain_view(self, imshape=None):
        """Get terrain camera view matrix."""
        if imshape is None:
            imshape = (self.resolution[1], self.resolution[0])
        return self.terrain_camera.get_view(imshape)

    def _toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        monitor = glfw.get_window_monitor(self.window)

        if monitor is None:
            # Currently windowed -> go fullscreen
            self.windowed_pos = glfw.get_window_pos(self.window)
            self.windowed_size = glfw.get_window_size(self.window)
            self.windowed_render_resolution = self.render_resolution

            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window,
                monitor,
                0,
                0,
                mode.size.width,
                mode.size.height,
                mode.refresh_rate,
            )
            self.resolution = (mode.size.width, mode.size.height)
            self.render_resolution = self.resolution  # Match display for sharpness
            self.fullscreen = True
        else:
            # Currently fullscreen -> go windowed
            if self.windowed_size is None:
                self.windowed_size = (1280, 720)
            if self.windowed_pos is None:
                self.windowed_pos = (100, 100)

            glfw.set_window_monitor(
                self.window,
                None,
                self.windowed_pos[0],
                self.windowed_pos[1],
                self.windowed_size[0],
                self.windowed_size[1],
                0,
            )
            self.resolution = self.windowed_size
            self.render_resolution = getattr(
                self, "windowed_render_resolution", self.resolution
            )
            self.fullscreen = False

        # Recreate FBOs at new render resolution
        self._recreate_fbos()

        # Update viewport and picker
        self._rebuild_viewports()
        if self.pyramid_picker is not None:
            self.pyramid_picker.resize(*self.resolution)

    def _recreate_fbos(self):
        """Recreate framebuffers at current render_resolution."""
        render_w, render_h = self.render_resolution

        # Release old FBOs
        if self.fbo_msaa is not None:
            self.fbo_msaa.release()
        if self.fbo is not None:
            self.fbo.release()

        # Create multisampled FBO for antialiased rendering
        self.fbo_msaa = self.ctx.framebuffer(
            color_attachments=[
                self.ctx.renderbuffer((render_w, render_h), 4, samples=self.samples)
            ],
            depth_attachment=self.ctx.depth_renderbuffer(
                (render_w, render_h), samples=self.samples
            ),
        )

        # Create resolved FBO with texture (for video encoding and display)
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((render_w, render_h), 4)],
            depth_attachment=self.ctx.depth_texture((render_w, render_h)),
        )
