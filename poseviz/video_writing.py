import os
import os.path as osp

def main_video_writer(q_out_video_frames):
    import imageio
    writer_kwargs = dict(codec='h264', ffmpeg_params=['-crf', '15'], macro_block_size=None)
    writer = None
    while True:
        frame = q_out_video_frames.get()
        if isinstance(frame, str) and frame == 'stop_video_writing':
            break

        # A tuple of a path and fps signifies that we need to start a new video
        if isinstance(frame, str) and frame == 'close_current_video':
            if writer is not None:
                writer.close()
                writer = None
        elif isinstance(frame, tuple):
            video_path, fps = frame
            if writer is not None:
                writer.close()
            os.makedirs(osp.dirname(video_path), exist_ok=True)
            writer = imageio.get_writer(video_path, fps=fps, **writer_kwargs)
        elif writer is not None:
            writer.append_data(frame)
        q_out_video_frames.task_done()

    if writer is not None:
        writer.close()
    q_out_video_frames.task_done()
