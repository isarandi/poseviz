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

When writing fails
------------------

If a requested video cannot be produced (encoder failure, unwritable path,
non-NVIDIA GL context with ``gpu_encode=True``), the error is raised in your
process as ``poseviz.VisualizerError`` at the next API call — ``update``,
``new_sequence_output``, ``finalize_sequence_output`` or ``close`` — carrying
the visualizer-side traceback. A requested video to which zero frames were
written also raises. The failure poisons only the current sequence: starting
a new sequence output resets the state, so a batch loop can catch the error
and continue with the next video::

    for clip in clips:
        try:
            viz.new_sequence_output(clip.out_path, fps=30)
            for frame in clip:
                viz.update(...)
            viz.finalize_sequence_output()
        except poseviz.VisualizerError as e:
            print(f'{clip}: {e}')

``close()`` returns one ``poseviz.SequenceReport`` per sequence (video path,
number of frames written, error if any), and a summary line is logged for
each successfully written video.

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
