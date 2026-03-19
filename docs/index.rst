PoseViz
=======

PoseViz is a 3D human pose visualization tool for multi-person, multi-camera
scenarios. It renders skeletons, body meshes, camera frustums, and images in
an interactive 3D scene.

.. image:: images/padding_0.2.jpg
   :width: 100%
   :align: center

Key features:

- **Multi-person**: Visualize multiple people with distinct skeletons and/or meshes
- **Multi-camera**: Display multiple camera views simultaneously
- **Body meshes**: Render SMPL and similar parametric body models
- **Non-blocking**: Visualization runs in a separate process, so your code
  stays in control of the main loop
- **Video output**: Record to video with GPU-accelerated encoding (NVENC)
- **Interactive navigation**: Orbit, pan, zoom, and fly through the scene

Quick start
-----------

::

    import poseviz
    import deltacamera

    camera = deltacamera.Camera(
        intrinsic_matrix=K,
        rot_world_to_cam=R,
        optical_center=t,
    )

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        for frame in video:
            boxes, poses = run_pose_estimation(frame)
            viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

Installation
------------

::

    pip install poseviz


.. toctree::
   :maxdepth: 2
   :caption: How-to Guides

   howto/usage
   howto/video-output
   howto/headless
   howto/multi-view
   howto/gpu-frames

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/architecture
   explanation/coordinates
   explanation/rendering

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
