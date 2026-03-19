Multi-View Visualization
========================

This guide covers how to visualize scenes captured from multiple cameras
simultaneously.

Basic multi-view usage
----------------------

Use ``update_multiview`` with a list of ``ViewInfo`` objects, one per camera::

    import poseviz
    import deltacamera

    with poseviz.PoseViz(joint_names, joint_edges, n_views=3) as viz:
        for frame1, frame2, frame3 in video_streams:
            view_infos = [
                poseviz.ViewInfo(
                    frame=frame1, boxes=boxes1, poses=poses1, camera=camera1,
                ),
                poseviz.ViewInfo(
                    frame=frame2, boxes=boxes2, poses=poses2, camera=camera2,
                ),
                poseviz.ViewInfo(
                    frame=frame3, boxes=boxes3, poses=poses3, camera=camera3,
                ),
            ]
            viz.update_multiview(view_infos)

Each camera appears in the 3D scene as a textured image quad (the camera's
view) plus a wireframe frustum showing its position and orientation.

Dynamic camera count
--------------------

The number of cameras can change from frame to frame. PoseViz automatically
creates or destroys view visualizers to match the number of ``ViewInfo``
objects passed::

    # Frame 1: two cameras
    viz.update_multiview([view_info_cam1, view_info_cam2])

    # Frame 2: three cameras
    viz.update_multiview([view_info_cam1, view_info_cam2, view_info_cam3])

The ``n_views`` constructor parameter only sets the initial allocation — it
does not limit the number of cameras.

Navigating between cameras
--------------------------

In the visualizer window:

- **1-9**: Jump to camera 1-9 (snaps the view to follow that camera)
- **n**: Cycle through cameras as the main view
- **d**: Snap to the nearest displayed camera
- **g**: Snap to the nearest camera and show only that camera's poses

When snapped to a camera, the view tracks that camera's position as it moves
through the sequence. Any manual camera movement (orbit, pan, fly) unsnaps
and returns to free navigation.

Controlling which poses are shown
---------------------------------

By default, poses from all cameras are shown simultaneously. This can get
cluttered with many cameras.

- **t**: Toggle between showing all poses or just the selected camera's poses
- **m**: Cycle which camera's poses are displayed (without changing the view)

Reducing visual clutter
-----------------------

Multi-camera scenes can get busy. Some useful settings::

    with poseviz.PoseViz(
        joint_names, joint_edges,
        show_field_of_view=False,       # Hide the extended frustum pyramids
        show_camera_wireframe=True,     # Keep the small camera wireframes
        camera_type='free',             # Start in free-fly mode to see all cameras
    ) as viz:
        ...

``show_field_of_view=False`` is particularly helpful — the extended frustum
lines from multiple cameras can quickly obscure the scene.

Omitting frames
---------------

If a camera has no image for a given timestep (e.g., dropped frame), pass
``frame=None`` in its ``ViewInfo``::

    poseviz.ViewInfo(frame=None, boxes=boxes, poses=poses, camera=camera)

When ``frame`` is None, both the image quad and camera wireframe are skipped
for that view. Poses are still rendered.
