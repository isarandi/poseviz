"""PoseViz is a 3D visualization tool for human pose estimation."""

from poseviz.poseviz import PoseViz
from poseviz.view_info import ViewInfo
from poseviz.errors import VisualizerError
from poseviz.messages import SequenceReport
from poseviz.scenefile import SceneReader, SceneWriter

try:
    from ._version import version as __version__
except ImportError:
    __version__ = '0.0.0'

__all__ = [
    "PoseViz",
    "ViewInfo",
    "VisualizerError",
    "SequenceReport",
    "SceneReader",
    "SceneWriter",
    "__version__",
]
