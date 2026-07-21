Coordinate Systems
==================

Points pass through several coordinate transformations on their way to the
screen. This page explains each space and the transformations between them.

Spaces
------

**World coordinates**: The user's coordinate system. Poses, camera positions,
and the ground plane are specified in world coordinates. The ``world_up``
parameter (default ``(0, -1, 0)``) tells PoseViz which direction is up.

**GL coordinates**: OpenGL uses a right-handed system with Y up, X right, and
Z toward the viewer (out of the screen). This differs from common conventions
in computer vision, where Y often points down and Z points away.

**Camera coordinates**: Camera-relative. The camera is at the origin, X points
right, Y points down, Z points forward (the OpenCV convention used by
deltacamera).

**Clip coordinates**: After projection. The projection matrix maps camera
coordinates to a normalized cube where visible points have coordinates in
[-1, 1]. The GPU clips geometry outside this range.

**Screen coordinates**: Pixel positions after the viewport transform.

The world up vector
-------------------

All geometry is rendered directly in world coordinates (millimeters); there is
no global re-orientation of the scene. Instead, the ``world_up`` parameter
determines everything that has a notion of "up":

- The **terrain camera** (interactive orbit/pan/fly navigation) builds an
  orthonormal basis from ``world_up`` via ``up_basis``; azimuth and elevation
  are spherical coordinates in that basis, so orbiting always revolves around
  the world's vertical axis.
- The **ground plane** is placed perpendicular to ``world_up``, at
  ``ground_plane_height`` along it.
- The view cameras created by the terrain camera use ``world_up`` for their
  roll orientation (via deltacamera's ``turned_towards``).

The basis construction from the up vector::

    forward = normalize(cross(up, rightlike))
    right = cross(forward, up)

For the default ``world_up=(0, -1, 0)`` (Y-down, standard in computer vision)
this yields ``right=(1, 0, 0)`` and ``forward=(0, 0, 1)``.

View and projection matrices
-----------------------------

When rendering from a specific camera (e.g., the ``original`` camera mode or
the image quad placement), PoseViz uses the camera's own intrinsics and
extrinsics to build the MVP matrix directly — a standard view matrix from the
camera's R and t, and a projection matrix from its intrinsic matrix K.

The view matrix transforms world coordinates to camera coordinates::

    view[:3, :3] = R
    view[:3,  3] = -R @ t

Where ``R`` is the world-to-camera rotation (rows are camera axes in world
coordinates) and ``t`` is the camera's optical center in world coordinates.

The projection matrix encodes the camera's intrinsics::

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

For image size ``(W, H)`` and depth range ``[near, far]``::

    [[ 2*fx/W,       0,      1 - 2*cx/W,                0             ],
     [ 0,           -2*fy/H, 2*cy/H - 1,                0             ],
     [ 0,            0,      (far+near)/(far-near),     -2*far*near/(far-near)],
     [ 0,            0,      1,                          0             ]]

Note the negative sign on ``2*fy/H``: this flips Y so the resulting FBO image
has row 0 at the top (matching video conventions), avoiding a CPU-side flip
during video encoding.

The fourth row sets ``w = Z`` (not ``-Z``), because the camera convention has
Z pointing forward (into the scene), so no additional negation is needed.

Combined MVP
------------

The model-view-projection (MVP) matrix combines all transformations::

    MVP = projection @ view

For PoseViz, there is no separate model matrix — geometry is already in
world coordinates. The combined matrix transforms world points directly to
clip coordinates in a single matrix multiply.

Before sending to the GPU, the matrix is transposed for GLSL's column-major
layout::

    return (projection @ view).T

In the shader::

    gl_Position = u_view_proj * vec4(world_position, 1.0);

The GPU then performs perspective division (``xyz / w``) and viewport
transformation to get final screen coordinates.

Depth handling
--------------

OpenGL's depth buffer expects values in [0, 1] after the viewport transform
(or [-1, 1] in clip space). The projection matrix maps the camera-space Z
range ``[near, far]`` to this interval.

Points closer than ``near`` or farther than ``far`` are clipped. Choosing
appropriate values matters:

- ``near`` too small: Depth precision suffers (Z-fighting artifacts)
- ``far`` too large: Same problem, precision spread too thin
- ``near`` too large: Nearby geometry gets clipped

PoseViz uses ``near=100.0`` and ``far=100000.0`` (in millimeters), suitable for
human-scale scenes viewed from typical camera distances.

The ground plane
----------------

The ground plane is positioned at ``ground_plane_height`` along the world's
up axis. For ``world_up=(0, -1, 0)``, this means the plane's Y coordinate
equals ``-ground_plane_height`` (since Y is flipped).

The checkerboard pattern is rendered as a textured quad. The texture
coordinates are computed in world XZ (or the horizontal plane for other
up vectors).
