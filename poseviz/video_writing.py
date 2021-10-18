import os


def main_video_writer(q_out_video_frames):
    import imageio
    writer_kwargs = dict(codec='h264', ffmpeg_params=['-crf', '18'], macro_block_size=None)
    writer = None

    while (frame := q_out_video_frames.get()) != 'stop_video_writing':
        # A tuple of a path and fps signifies that we need to start a new video
        if isinstance(frame, tuple):
            video_path, fps = frame
            if writer is not None:
                writer.close()
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            writer = imageio.get_writer(video_path, fps=fps, **writer_kwargs)
            q_out_video_frames.task_done()
        elif writer is not None:
            writer.append_data(frame)
            q_out_video_frames.task_done()

    if writer is not None:
        writer.close()
    q_out_video_frames.task_done()
