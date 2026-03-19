# PoseViz – 3D Human Pose and Mesh Visualizer

<p align="center">
  <img src=screenshot.jpg width="30%">
  <img src=screenshot2.jpg width="30%">
  <img src=screenshot_multicam.jpg width="30%">
</p>

Multi-person, multi-camera 3D human pose and mesh visualization tool built using
OpenGL (via ModernGL). As used in [NLF](https://github.com/isarandi/nlf) and [MeTRAbs](https://github.com/isarandi/metrabs) visualizations.

**This repo does not contain pose estimation code, only the visualization part.**

## Gist of Usage

```python
import poseviz
import deltacamera

camera = deltacamera.Camera(...)

with poseviz.PoseViz(...) as viz:
    for frame in frames:
        bounding_boxes, poses3d = run_pose_estimation_model(frame)
        viz.update(frame=frame, boxes=bounding_boxes, poses=poses3d, camera=camera)
```

See also [```demo.py```](demo.py).

The main feature of this tool is that the graphical event loop is hidden from the library user. We
want to write code in terms of the *prediction loop* of the human pose estimator, not from the point
of view of the visualizer tool.

Behind the scenes, this is achieved through forking a dedicated visualization process and passing
new scene information via multiprocessing queues.

## Installation

```bash
pip install poseviz
```

Then run [demo.py](demo.py) to test if installation was successful.

