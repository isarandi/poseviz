Using GPU Frames
================

This guide covers how to feed GPU tensors directly to PoseViz, avoiding
CPU round-trips when your frames are already on the GPU.

Enable GPU frame mode
---------------------

Set ``gpu_frames=True`` when creating the visualizer::

    with poseviz.PoseViz(
        joint_names, joint_edges,
        gpu_frames=True,
    ) as viz:
        ...

This changes how frames are transferred between the main process and the
visualizer process. Instead of shared-memory ring buffers (the default CPU
path), frames are passed via PyTorch's CUDA IPC mechanism.

Pass GPU tensors to update
--------------------------

With ``gpu_frames=True``, pass GPU tensors as the ``frame`` argument::

    import torch

    with poseviz.PoseViz(..., gpu_frames=True) as viz:
        for frame_tensor in video_decoder:
            # frame_tensor: CUDA tensor, shape (H, W, 3), dtype uint8
            viz.update(frame=frame_tensor, camera=camera, ...)

The frame must be:

- Shape ``(H, W, 3)`` — height, width, RGB channels
- dtype ``uint8``
- On a CUDA device
- Contiguous (non-contiguous tensors are made contiguous automatically, but
  this costs a copy)

Any DLPack-compatible GPU object is accepted — PyTorch CUDA tensors, CuPy
arrays, JAX arrays, hardware video decoder outputs, etc. Non-PyTorch objects
are converted to PyTorch tensors internally via ``torch.from_dlpack``.

What happens on the GPU path
-----------------------------

When ``gpu_frames=True``, PoseViz performs image downscaling and undistortion
on the GPU using ``deltacamera.pt.reproject_image``, which runs a single
``grid_sample`` kernel. This replaces the CPU-based OpenCV resize and remap
that the default path uses.

The processed tensor is then transferred to the visualizer process via
PyTorch's CUDA IPC (inter-process communication) and uploaded to an OpenGL
texture for rendering.

What you lose
-------------

2D overlays (bounding boxes and 2D pose skeletons drawn directly on the image)
require a NumPy array and are **skipped** when frames are GPU tensors. The 3D
rendering (skeletons, meshes, ground plane) is unaffected — only the painted-on
2D annotations on the camera image are missing.

If you need 2D overlays, use the default CPU path (``gpu_frames=False``) and
pass NumPy arrays.

Example: GPU video decoder to visualization
--------------------------------------------

A typical GPU-native pipeline using ``framepump`` for hardware video decoding::

    import torch
    import framepump
    import poseviz

    video = framepump.VideoFramesCuda('input.mp4', output_format='rgb')

    with poseviz.PoseViz(
        joint_names, joint_edges,
        gpu_frames=True,
        gpu_encode=True,
        out_video_path='output.mp4',
        out_fps=video.fps,
    ) as viz:
        for frame_dlpack in video:
            frame = torch.from_dlpack(frame_dlpack)
            viz.update(frame=frame, camera=camera, ...)

This keeps the entire pipeline on the GPU: decode (NVDEC) -> resize/undistort
(CUDA) -> render (OpenGL) -> encode (NVENC).
