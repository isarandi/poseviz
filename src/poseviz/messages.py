from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UpdateScene:
    view_infos: list = field(default_factory=list)
    viz_camera: object = None
    viz_imshape: Optional[tuple] = None


@dataclass
class AppendFrame:
    frame: object = None


@dataclass
class StartSequence:
    video_path: Optional[str] = None
    fps: int = 30
    resolution: Optional[tuple] = None
    camera_trajectory_path: Optional[str] = None
    audio_source_path: Optional[str] = None


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
    ringbuffers: list = field(default_factory=list)
