# API Reference

## poseviz.PoseViz

The main class that creates a visualizer process and can then be used as the interface to update the
visualization state.

This class supports the Python **context manager** protocol to close the visualization at the end.

### Constructor

```python
poseviz.PoseViz(
    joint_names, joint_edges, camera_type='free', n_views=1, world_up=(0, -1, 0),
    ground_plane_height=-1000, downscale=1, viz_fps=100, queue_size=64, write_video=False,
    multicolor_detections=False, snap_to_cam_on_scene_change=True, high_quality=True,
    draw_2d_pose=False, show_field_of_view=True, resolution=(1280, 720),
    use_virtual_display=False, show_virtual_display=True
)
```

#### Arguments:
- **joint_names**: an iterable of strings, the names of the human body joints that will be
  visualized. Left joints must start with 'l', right joints with 'r', mid-body joints with something
  else. Currently only the first character is inspected in each joint name, to color the left, right
  and middle part of the stick figure differently.
- **joint_edges**: an iterable of joint-index pairs, describing the bone-connectivity of the stick
  figure.
- **camera_type**: 
- **n_views**: Integer, the number of cameras that will be displayed.
- **world_up**: A 3-vector, the up vector in the world coordinate system in which the poses will be
  specified.
- **ground_plane_height**: Scalar, the vertical position of the ground plane in the world coordinate
  system.
- **downscale**: Integer, an image downscaling factor for display. May speed up the visualization.
- **viz_fps**: Target frames-per-second of the visualization. If the updates would come faster than
  this, the visualizer will block, to ensure that visualization does not happen faster than this
  FPS. Of course if the speed of updates cannot deliver this fps, the visualization will also be
  slower.
- **queue_size**: Integer, size of the internal queue used to communicate with the visualizer
  process.
- **write_video**: Boolean, whether the visualization output will be recorded to a video file.
- **multicolor_detections**: Boolean, color each box with a different color. Useful when tracking
  with fixed IDs.
- **snap_to_cam_on_scene_change**: Boolean, whether to reinitialize the view camera to the original
  camera on each change of sequence (call to ```viz.reinit_camera_view()```).
- **high_quality**: Boolean, whether to use high resolution spheres and tubes for the skeletons (may
  be faster to set it to False).
- **draw_2d_pose**: Boolean, whether to draw the 2D skeleton on the displayed camera image.
- **show_field_of_view**: Boolean, whether to visualize an extended pyramid indicating what is part
  of the field of view of the cameras. Recommended to turn off in multi-camera setups, as the
  visualization can get crowded otherwise.
- **resolution**: The resolution of the visualization window (width, height) integer pair.
- **use_virtual_display**: Boolean, whether to use a virtual display for visualization. There may be
  two reasons to do this. First, to allow higher resolution visualization than the screen
  resolution. Normally, Mayavi won't allow windows that are larger than the display screen (just
  automatically "maximizes" the window.). Second, it can be a way to do off-screen rendering.
- **show_virtual_display**: Boolean, whether to show the virtual display or to hide it (off-screen
  rendering). Has no effect if `use_virtual_display``` is False.


### update

```python
viz.update(frame, boxes, poses, camera, poses_true=(), poses_alt=(), block=True)
```

#### Arguments:

-**frame**
-**boxes**
-**poses**
-**camera**


#### Return value:


### update

```python
viz.update_multiview(view_infos: List[ViewInfo], block=True)
```

#### Arguments:


#### Return value:


### update

```python
viz.reinit_camera_view()
```

#### Arguments:


#### Return value:


### update

```python
viz.new_sequence_output(new_video_path, fps)
```

#### Arguments:

#### Return value:


### update

```python
viz.end_sequence()
```

#### Arguments:

#### Return value:


## poseviz.Camera

Class for specifying where to visualize the cameras and how to project poses onto them.

### Constructor

```python
poseviz.Camera(
    optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
    distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None)
)
```

#### Arguments:

- **optical_center**: The position of the camera as a 3-vector.
- **rot_world_to_cam**: 3x3 matrix of rotation from world coordinate system to camera system
- **intrinsic_matrix**: 3x3 matrix of the intrinsics
- **distortion_coeffs**: a vector of 5 elements, radial and tangential distortion coefficients
  according
  to [OpenCV's order](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)
  (k1, k2, p1, p2, k3).
- **world_up**: Up vector in the world (3-vector)
- **extrinsic_matrix**: 4x4 matrix, an alternative way of specifying extrinsics instead of using `
  optical center` and `rot_world_to_cam`.
  
