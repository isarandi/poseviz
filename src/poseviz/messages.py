from dataclasses import dataclass


@dataclass
class UpdateScene:
    view_infos: any
    viz_camera: any
    viz_imshape: any


@dataclass
class AppendFrame:
    frame: any


@dataclass
class StartSequence:
    video_path: any =None
    fps: any = 30
    resolution: any = None
    camera_trajectory_path: any = None
    audio_source_path: any = None


class EndSequence:
    pass


class Pause:
    pass


class Resume:
    pass


class ReinitCameraView:
    pass


class Quit:
    pass


class Nothing:
    pass


@dataclass
class NewRingBuffers:
    ringbuffers: any
