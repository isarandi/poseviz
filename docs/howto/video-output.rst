Recording Video Output
======================

This guide covers how to save visualizations to video files.

Record from the start
---------------------

Pass ``out_video_path`` and ``out_fps`` when creating the visualizer::

    with poseviz.PoseViz(
        joint_names, joint_edges,
        out_video_path='output.mp4',
        out_fps=30,
    ) as viz:
        for frame in video:
            viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

The video is finalized automatically when the context manager exits.

Start and stop recording mid-session
-------------------------------------

Use ``new_sequence_output`` and ``finalize_sequence_output`` to control
recording within a session::

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        # ... visualize some frames without recording ...

        viz.new_sequence_output('segment_1.mp4', fps=30)
        for frame in segment_1:
            viz.update(...)
        viz.finalize_sequence_output()

        viz.new_sequence_output('segment_2.mp4', fps=25)
        for frame in segment_2:
            viz.update(...)
        viz.finalize_sequence_output()

You can record multiple segments to different files within a single session.

Copy audio from a source file
------------------------------

To copy the audio track from the original video (or any audio file) into the
output::

    with poseviz.PoseViz(
        ...,
        out_video_path='output.mp4',
        out_fps=30,
        audio_path='source_video.mp4',
    ) as viz:
        ...

Or when starting mid-session::

    viz.new_sequence_output('output.mp4', fps=30, audio_source_path='source.mp4')

GPU vs CPU encoding
-------------------

By default, PoseViz uses GPU-accelerated encoding via NVENC (``gpu_encode=True``).
This avoids reading pixels back to the CPU entirely — the rendered framebuffer
texture is passed directly to the hardware encoder through CUDA-OpenGL interop.

To fall back to CPU encoding (e.g., on machines without NVENC)::

    with poseviz.PoseViz(..., gpu_encode=False) as viz:
        ...

With CPU encoding, pixels are read back from the GPU each frame, which is
significantly slower.

Control output resolution
-------------------------

By default, the video is rendered at the window's display resolution. To
render at a higher resolution (e.g., for publication-quality output), set
``render_resolution``::

    with poseviz.PoseViz(
        ...,
        resolution=(1280, 720),          # Window size
        render_resolution=(1920, 1080),   # Video output size
        out_video_path='hires.mp4',
        out_fps=30,
    ) as viz:
        ...

The window shows a downscaled version while the video is encoded at the full
render resolution.
