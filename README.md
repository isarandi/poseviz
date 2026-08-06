# PoseViz – 3D Human Pose and Mesh Visualizer

<p align="center">
  <img src=screenshot.jpg width="30%">
  <img src=screenshot2.jpg width="30%">
  <img src=screenshot_multicam.jpg width="30%">
</p>

Multi-person, multi-camera 3D human pose and mesh visualization tool built using
OpenGL (via ModernGL). As used in [NLF](https://github.com/isarandi/nlf) and [MeTRAbs](https://github.com/isarandi/metrabs) visualizations.

**This repo does not contain pose estimation code, only the visualization part.**

## Gist of Usage

```python
import poseviz
import deltacamera

camera = deltacamera.Camera(...)

with poseviz.PoseViz(...) as viz:
    for frame in frames:
        bounding_boxes, poses3d = run_pose_estimation_model(frame)
        viz.update(frame=frame, boxes=bounding_boxes, poses=poses3d, camera=camera)
```

See also [```demo.py```](demo.py).

The main feature of this tool is that the graphical event loop is hidden from the library user. We
want to write code in terms of the *prediction loop* of the human pose estimator, not from the point
of view of the visualizer tool.

Behind the scenes, this is achieved through forking a dedicated visualization process and passing
new scene information via multiprocessing queues.

## Error Handling

When a requested output cannot be produced (the video writer fails, the camera trajectory
cannot be saved, or the visualizer process dies), a `poseviz.VisualizerError` is raised in
your prediction loop at the next API call, carrying the visualizer-side traceback. A
requested video output to which zero frames were written also raises. Purely visual
per-frame hiccups do not raise — they are logged, and only escalate to an error if they
persist while an output is being written. If no output is requested (live demo use), the
visualization stays best-effort and never interrupts your loop.

`close()` returns a list of `poseviz.SequenceReport` objects, one per output sequence,
with the video path and the number of frames written.

## Headless and GPU Notes

- `headless=True` (or auto-detection when `DISPLAY`/`WAYLAND_DISPLAY` are unset) uses a
  standalone EGL context: no display server is needed, and on hybrid-GPU machines
  (e.g., AMD display GPU + NVIDIA compute GPU) the NVIDIA card is targeted for NVENC.
  There is no need to unset `DISPLAY` manually; the `egl_device_index` parameter
  overrides the device selection if needed.
- `gpu_encode=True` (default) requires the OpenGL context to be on an NVIDIA GPU. In
  windowed mode on hybrid-GPU machines, the window usually lands on the display GPU;
  starting a video output then raises an error immediately, listing the remedies: start
  the process with `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia`, use
  headless mode, or pass `gpu_encode=False` for CPU encoding.
- `gpu_frames=True` accepts PyTorch CUDA tensors or any DLPack-compatible frames (CuPy,
  JAX, ...), keeping the whole decode → inference → visualization → encode chain on the
  GPU.

## Installation

```bash
pip install poseviz
```

Then run [demo.py](demo.py) to test if installation was successful.

