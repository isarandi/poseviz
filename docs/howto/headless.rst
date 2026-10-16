Headless Rendering
==================

This guide covers how to run PoseViz without a display, for batch processing
or server environments.

Automatic headless detection
----------------------------

By default (``headless=None``), PoseViz checks for a display server
automatically. On Linux, if neither ``DISPLAY`` (X11) nor ``WAYLAND_DISPLAY`` is
set, it switches to headless mode. On macOS and Windows, which do not use these
variables, it always opens a window. This means code that works on a desktop also
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

Headless mode obtains its OpenGL context directly from EGL (standalone, no
display server involved) and renders to an offscreen framebuffer. This works
regardless of whether ``DISPLAY`` is set; if EGL is unavailable, an invisible
window is used as fallback. No window is ever shown on screen.

On multi-GPU machines (e.g. a hybrid desktop with an AMD display GPU and an
NVIDIA compute GPU), the EGL devices are searched for an NVIDIA GPU unless
``gpu_encode=False``, so NVENC encoding works no matter which GPU drives the
desktop. Pass ``egl_device_index`` to select the EGL device manually.

Headless mode is typically combined with video recording — without it, the
rendered frames have nowhere to go.

GPU-accelerated pipeline
------------------------

With ``headless=True``, the full pipeline can stay on the GPU:

1. OpenGL renders the scene to a framebuffer object (FBO)
2. The FBO texture is passed to NVENC via CUDA-OpenGL interop
3. The encoded video is written to disk

No pixel data is ever copied to the CPU. This is the fastest path for batch
rendering. It needs an NVIDIA GPU and framepump's CUDA bindings::

    pip install 'framepump[cuda]'

CPU fallback
------------

With the default ``gpu_encode=None``, PoseViz uses the zero-copy path only when
it can work here (EGL context on an NVIDIA GPU, CUDA bindings installed). Otherwise
it logs the reason, reads the pixels back from the GPU each frame and encodes them
with ``framepump.VideoWriter`` (NVENC from system memory where available, else the
CPU). Headless video output therefore works on any machine with OpenGL support,
also without an NVIDIA GPU.

``gpu_encode=True`` insists on the zero-copy path and raises an error naming the
remedies when it cannot be used; ``gpu_encode=False`` always reads back and encodes
on the CPU::

    with poseviz.PoseViz(
        ...,
        headless=True,
        gpu_encode=False,
        out_video_path='output.mp4',
        out_fps=30,
    ) as viz:
        ...

Choosing the output resolution
------------------------------

In headless mode, ``render_resolution`` controls the actual rendering size;
``resolution`` serves only as its default when ``render_resolution`` is not
set::

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
