# PoseViz – 3D Human Pose Visualizer

<p align="center">
  <img src=screenshot.jpg width="30%">
  <img src=screenshot2.jpg width="30%">
  <img src=screenshot_multicam.jpg width="30%">
</p>

Multi-person, multi-camera 3D human pose visualization tool built using
[Mayavi](https://docs.enthought.com/mayavi/mayavi/). As used
in [MeTRAbs](https://github.com/isarandi/metrabs) visualizations.

## Gist of Usage

```python
import poseviz

viz = poseviz.PoseViz(...)
camera = poseviz.Camera(...)
for frame in frames:
    bounding_boxes, poses3d = run_pose_estimation_model(frame)
    viz.update(frame=frame, boxes=bounding_boxes, poses=poses3d, camera=camera)
```

The main feature of this tool is that the graphical event loop is hidden from the library user. We
want to write code in terms of the *prediction loop* of the human pose estimator, not from the point
of view of the visualizer tool.

Behind the scenes, this is achieved through forking a dedicated visualization process and passing
new scene information via multiprocessing queues.

## Installation

PoseViz is released as a conda package (experimental, tested only on Linux):

```bash
conda install poseviz -c isarandi
```

Alternatively, in case the above doesn't work, install Mayavi via conda (the Mayavi pip package has
compilation problems), clone this repo and install PoseViz via pip:

```bash
conda install mayavi -c conda-forge
pip install git+https://github.com/isarandi/poseviz.git
```