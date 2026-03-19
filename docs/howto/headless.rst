Headless Rendering
==================

This guide covers how to run PoseViz without a display, for batch processing
or server environments.

Automatic headless detection
----------------------------

By default (``headless=None``), PoseViz checks for a display server
automatically. If neither ``DISPLAY`` (X11) nor ``WAYLAND_DISPLAY`` is set,
it switches to headless mode. This means code that works on a desktop also
works on a remote server without changes::

    with poseviz.PoseViz(
        joint_names, joint_edges,
        out_video_path='output.mp4',
        out_fps=30,
    ) as viz:
        for frame in video:
            viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

On a desktop this opens a window; on a headless server it renders offscreen
and writes the video.

You can override the auto-detection by passing ``headless=True`` or
``headless=False`` explicitly.

Headless mode creates an invisible GLFW window to obtain an OpenGL context,
then renders to an offscreen framebuffer. No window is ever shown on screen.

Headless mode is typically combined with video recording — without it, the
rendered frames have nowhere to go.

GPU-accelerated pipeline
------------------------

With ``headless=True`` and ``gpu_encode=True`` (the latter is the default),
the full pipeline stays on the GPU:

1. OpenGL renders the scene to a framebuffer object (FBO)
2. The FBO texture is passed to NVENC via CUDA-OpenGL interop
3. The encoded video is written to disk

No pixel data is ever copied to the CPU. This is the fastest path for batch
rendering.

CPU fallback
------------

On machines without NVENC support::

    with poseviz.PoseViz(
        ...,
        headless=True,
        gpu_encode=False,
        out_video_path='output.mp4',
        out_fps=30,
    ) as viz:
        ...

This reads pixels back from the GPU each frame and encodes on the CPU.
Significantly slower, but works on any machine with OpenGL support.

Choosing the output resolution
------------------------------

In headless mode, ``resolution`` controls the size of the invisible window
(used as a fallback), while ``render_resolution`` controls the actual
rendering size::

    with poseviz.PoseViz(
        ...,
        headless=True,
        render_resolution=(1920, 1080),
        out_video_path='output.mp4',
        out_fps=30,
    ) as viz:
        ...

If ``render_resolution`` is not set, it defaults to ``resolution``.

Remote servers (SSH)
--------------------

Over SSH without X forwarding, ``DISPLAY`` is unset, so PoseViz automatically
enters headless mode. It only requires a GPU with OpenGL and EGL support, which
is standard on modern NVIDIA drivers.
