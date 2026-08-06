from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UpdateScene:
    view_infos: list = field(default_factory=list)
    viz_camera: object = None
    viz_imshape: Optional[tuple] = None


@dataclass
class StartSequence:
    video_path: Optional[str] = None
    fps: int = 30
    camera_trajectory_path: Optional[str] = None
    audio_source_path: Optional[str] = None


@dataclass
class Fault:
    """Failure notification sent from the visualizer process to the producer.

    The producer raises a VisualizerError built from this at its next API call.
    """

    seq_index: Optional[int]  # None for faults not tied to a sequence
    context: str  # what was being attempted when the failure occurred
    error_type: str
    error_message: str
    traceback_str: str = ""


@dataclass
class SequenceReport:
    """Summary of one output sequence, sent when the sequence ends.

    Purely informational: errors are raised via Fault messages, not from
    reports. Collected reports are returned by PoseViz.close().
    """

    seq_index: int
    video_path: Optional[str]
    camera_trajectory_path: Optional[str]
    frames_written: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None


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
