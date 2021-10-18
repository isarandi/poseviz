# PoseViz – 3D Human Pose Visualizer

<p align="center"><img src=screenshot.jpg width="60%"></p>
<p align="center"><img src=screenshot2.jpg width="60%"></p>
<p align="center"><img src=screenshot_multicam.jpg width="60%"></p>

Multi-person, multi-camera 3D human pose visualization tool built using
[Mayavi](https://docs.enthought.com/mayavi/mayavi/). As used
in [MeTRAbs](https://github.com/isarandi/metrabs) visualizations.

## Gist of usage

```python
import poseviz

viz = poseviz.PoseViz(...)
camera = poseviz.Camera(...)
for frame in frames:
    bounding_boxes, poses3d = run_pose_estimation_model(frame)
    viz.update(frame=frame, boxes=bounding_boxes, poses=poses3d, camera=camera)
```

Detailed docs TBA.

## Installation

Install Mayavi via Conda (the Mayavi pip package has compilation problems), clone this repo and
install PoseViz via pip.

```bash
conda install mayavi -c conda-forge
pip install .
```

Then run [demo.py](demo.py) to test if installation was successful.
