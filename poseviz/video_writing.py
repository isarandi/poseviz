import os
import os.path as osp
import queue
import threading

import imageio.v2 as imageio
from poseviz import messages


def main_video_writer(q):
    writer = None
    while True:
        msg = q.get()

        if isinstance(msg, messages.AppendFrame) and writer is not None:
            writer.append_data(msg.frame)
        elif isinstance(msg, messages.StartSequence):
            if writer is not None:
                writer.close()

            os.makedirs(osp.dirname(msg.video_path), exist_ok=True)
            writer = imageio.get_writer(
                msg.video_path, codec='h264', input_params=['-r', str(msg.fps)],
                output_params=['-crf', '15'], macro_block_size=None)
        elif isinstance(msg, (messages.EndSequence, messages.Quit)):
            if writer is not None:
                writer.close()
                writer = None

        q.task_done()

        if isinstance(msg, messages.Quit):
            return


class VideoWriter:
    def __init__(self, queue_size=32):
        self.q = queue.Queue(queue_size)
        self.thread = threading.Thread(target=main_video_writer, args=(self.q,), daemon=True)
        self.thread.start()
        self.active = False

    def start_sequence(self, video_path, fps):
        self.q.put(messages.StartSequence(video_path, fps))
        self.active = True

    def is_active(self):
        return self.active

    def append_data(self, frame):
        self.q.put(messages.AppendFrame(frame))

    def end_sequence(self):
        self.q.put(messages.EndSequence())
        self.active = False

    def finish(self):
        self.q.put(messages.Quit())
        self.q.join()
        self.thread.join()
        self.active = False
