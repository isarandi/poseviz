Rendering
=========

This page explains how PoseViz builds and renders the 3D scene: what objects
exist, how they're organized, and how they get drawn efficiently.

What gets rendered
------------------

A PoseViz scene contains:

- **Displayed cameras**: Each camera from the input data appears as a textured
  quad (the image) plus a wireframe pyramid (the frustum). Multiple cameras
  can be shown simultaneously.

- **Skeletons**: 3D poses rendered as spheres (joints) connected by tubes
  (limbs). Different colors distinguish left/right/mid body parts and
  predicted/ground-truth/alternative poses (the latter is useful for comparing two algorithm's outputs at the same time against a ground truth).

- **Body meshes**: SMPL or similar parametric body models, rendered as
  triangle meshes with per-vertex normals for lighting.

- **Ground plane**: A checkerboard plane at a configurable height, providing
  spatial reference.

Scene graph structure
---------------------

The renderer organizes objects in a simple hierarchy::

    PoseVizGLSide (main renderer)
    ├── ViewVisualizer (one per displayed camera)
    │   ├── ImageQuadRenderable (textured quad for camera image)
    │   ├── WireframeRenderable (camera frustum pyramid)
    │   ├── SphereRenderable × 5 (joints: pred left/mid/right, true, alt)
    │   ├── TubeRenderable × 5 (limbs: pred left/mid/right, true, alt)
    │   └── MeshRenderable × N (one per body mesh)
    ├── GroundPlaneRenderable
    ├── PyramidPicker (for click-to-select camera)
    └── TerrainCamera (interactive view camera)

``PoseVizGLSide`` owns the OpenGL context, framebuffers, and main loop. It
receives messages from the main process, updates the visualizers, and issues
draw calls.

``ViewVisualizer`` groups everything related to one displayed camera. When a
new frame arrives, it updates the image texture, skeleton positions, and mesh
vertices for that camera.

Renderable primitives
---------------------

Each renderable type handles its own geometry, shaders, and GPU buffers:

**SphereRenderable**: Renders joints as spheres. Uses GPU instancing—one draw
call renders all spheres of the same color. The instance buffer stores
``(x, y, z, scale)`` per sphere.

**TubeRenderable**: Renders limbs as cylinders connecting two points. Also
instanced, with ``(start_x, start_y, start_z, end_x, end_y, end_z, radius)``
per tube.

**MeshRenderable**: Renders triangle meshes (body models). Vertices update each
frame; topology (faces) stays fixed. Computes per-vertex normals from faces
for Phong shading.

**ImageQuadRenderable**: Renders the camera's image as a textured quad
positioned in 3D space at ``image_plane_distance`` from the camera origin.

**WireframeRenderable**: Renders line segments. Used for camera frustum pyramids
but can render arbitrary wireframes.

**GroundPlaneRenderable**: A large textured quad with a procedural checkerboard
pattern.

Instanced rendering
-------------------

Rendering thousands of spheres (joints) or tubes (limbs) with individual draw
calls could be slow. PoseViz uses instancing: upload all instance data to a
buffer, then render with a single draw call.

For spheres, the vertex shader receives the unit sphere geometry plus
per-instance position and scale::

    in vec3 in_position;      // Unit sphere vertex
    in vec3 in_normal;
    in vec3 instance_pos;     // Per-instance
    in float instance_scale;

    uniform mat4 u_view_proj;

    void main() {
        vec3 world_pos = in_position * instance_scale + instance_pos;
        gl_Position = u_view_proj * vec4(world_pos, 1.0);
    }

The CPU uploads a buffer of ``(x, y, z, scale)`` tuples, and the GPU renders
all spheres in one batch.

Skeleton coloring
-----------------

Skeletons use color to distinguish body sides and pose types.

**Side classification**: Joint names starting with ``'l'`` are left side
(blue), ``'r'`` are right side (yellow), others are mid-body (cyan for joints,
green for limbs). An edge is classified as left only if both endpoint joints
start with ``'l'``, right only if both start with ``'r'``, and mid otherwise.

**Pose types**: Predicted poses use the left/mid/right coloring. Ground-truth
poses are red. Alternative poses are orange.

This requires five sphere renderers and five tube renderers per
``ViewVisualizer``: three for predicted (left/mid/right), one for true, one
for alt, times two for spheres vs tubes.

Image plane rendering
---------------------

Each displayed camera's image appears as a textured quad floating in 3D space.
The quad is positioned at ``image_plane_distance`` (default 1000mm) in front
of the camera.

To compute the quad corners, we unproject the image corners to world
coordinates::

    image_corners = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
    world_corners = camera.image_to_world(image_corners, depth=image_plane_distance)

The image is uploaded as a texture each frame. OpenGL expects textures with
Y=0 at the bottom, so the image is flipped during upload.

The quad renders with alpha blending (50% opacity by default) so the skeleton
behind it remains visible. Depth writes are disabled during image rendering
so that multiple semi-transparent image planes don't occlude each other—if
one camera's image wrote to the depth buffer, a second camera's image behind
it would fail the depth test and not render, even though both are transparent.

Mesh rendering
--------------

Body meshes (SMPL, SMPL-X, etc.) share a fixed face topology but have
per-frame vertex positions. ``MeshRenderable`` handles this by:

1. Pre-allocating vertex and normal buffers sized for the maximum expected
   vertex count
2. Uploading new vertex positions each frame
3. Recomputing per-vertex normals from the faces

Normal computation averages face normals at each vertex::

    # Face normals from cross product
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = cross(v1 - v0, v2 - v0)

    # Accumulate at vertices
    normals = zeros_like(vertices)
    add.at(normals, faces[:, 0], face_normals)
    add.at(normals, faces[:, 1], face_normals)
    add.at(normals, faces[:, 2], face_normals)
    normals = normalize(normals)

The ``ColorSource`` abstraction allows different coloring strategies without
changing the mesh renderer:

- ``UniformColor``: Single color for the entire mesh
- ``VertexRGBColor``: Per-vertex RGB values
- ``ScalarColormapColor``: Per-vertex scalar mapped through a colormap
- ``TextureColor``: UV-mapped texture

Each color source selects its shader variant and manages its GPU resources.

The render loop
---------------

Each frame follows this sequence:

1. **Receive message**: Get ``UpdateScene`` from the queue (or ``Nothing`` if
   paused)

2. **Update visualizers**: Reconstruct frames from shared memory, update
   skeletons, meshes, and image textures

3. **Render to MSAA FBO**: Draw the scene to a multisampled framebuffer (4x
   MSAA) at ``render_resolution``. This may be larger than the window for
   high-resolution video output.

4. **Resolve**: Copy the multisampled FBO to a regular FBO, resolving the
   antialiasing

5. **Blit to screen**: Draw the resolved FBO to the window, scaling if
   ``render_resolution`` differs from window size

6. **Capture frame**: If recording video, either pass the FBO texture directly
   to GLVideoWriter (zero-copy NVENC encoding) or read pixels back to CPU

The Y-flip deserves attention. Video formats expect row 0 to be the top of
the image, but OpenGL's default is Y=0 at the bottom. Rather than flipping
pixels on readback (which would prevent zero-copy encoding), we flip the
scene during rendering: the view-projection matrix is right-multiplied by
``diag(1, -1, 1, 1)``, which negates the Y component of all input world
coordinates, effectively rendering the scene upside-down into the FBO. This
makes the FBO texture directly usable by NVENC without any CPU-side copying
or flipping.

For window display, the fullscreen blit quad has flipped texture coordinates
(V=1 at screen bottom, V=0 at top), so the scene appears right-side-up.

Lighting
--------

All lit objects (spheres, tubes, meshes) use the same lighting setup: a
Raymond 3-point rig. The lights are
**camera-relative**—they move with the view camera, so the scene always has
consistent illumination regardless of the viewing angle.

The three directional lights, defined in camera space:

- **Key light** ``(0.5, -0.7071, -0.5)``: Front-right, from above. Full
  intensity. This is the dominant light source.

- **Fill light** ``(-0.75, 0.5, -0.433)``: Front-left, from below. 60%
  intensity. Softens the shadows cast by the key light.

- **Rim fill** ``(0.75, 0.5, -0.433)``: Front-right, from below. 50%
  intensity. Adds subtle definition on the opposite side.

(Camera-space convention: +X = right, +Y = down, −Z = forward into the scene.)

The shading model is pure Lambertian diffuse (``max(dot(n, l), 0)``) with no
specular component, plus a small ambient term of 0.1::

    float diffuse = max(dot(n, l0), 0.0)
                  + max(dot(n, l1), 0.0) * 0.6
                  + max(dot(n, l2), 0.0) * 0.5;
    float intensity = min(0.1 + diffuse, 1.0);

The light directions are transformed from camera space to world space each
fragment using ``transpose(mat3(u_view))``, so they rotate with the camera
automatically. This setup is duplicated identically across all six fragment
shaders (sphere, tube, mesh, mesh_textured, mesh_vertexcolor, mesh_scalar).

Camera controls
---------------

The view camera (what you see the scene through) can operate in three modes:

**Original mode**: The view follows one of the displayed cameras. As the
displayed camera moves through the video, your view moves with it.

**Bird mode**: An elevated view, positioned above and behind the displayed
camera, looking down at the scene. Useful for seeing the spatial relationship
between camera and subject.

**Free mode**: A ``TerrainCamera`` provides interactive navigation. It
maintains:

- A pivot point (center of rotation)
- Azimuth and elevation angles (orbit position)
- Distance from pivot
- Field of view

Mouse controls:

- Left drag: Orbit around pivot (camera moves, pivot stays fixed)
- Shift + left drag: Look around from current position (camera stays fixed,
  pivot moves to match the new viewing direction)
- Middle drag: Pan (move pivot parallel to view plane)
- Right drag / scroll: Zoom (change distance from pivot)

Keyboard controls:

- Arrow keys: Fly forward/back/left/right
- Page Up/Down: Fly up/down
- +/-: Adjust field of view

The terrain camera can also "snap" to a displayed camera, meaning it tracks
that camera's position while still allowing the user to orbit around it.
Pressing a number key (1-9) snaps to that camera. User movement unsnaps and
returns to free navigation.

A history stack enables back/forward navigation between camera positions,
similar to a web browser. Mouse button 4/5 (back/forward) navigate the
history.

Split-screen and viewports
--------------------------

Pressing Tab toggles split-screen mode, which divides the window into two
viewports: the left half shows the original camera, the right half shows the
free-fly terrain camera.

The ``Viewport`` dataclass (``src/poseviz/gl/viewport.py``) represents a
screen region with its own view-projection matrix::

    @dataclass
    class Viewport:
        name: str                        # "original" or "terrain"
        bounds: tuple                    # (x, y, width, height) in pixels
        get_view_proj: Callable          # Returns view-projection matrix
        get_matrices: Callable = None    # Returns (view_proj, view) tuple
        interactive: bool = False        # Receives mouse/keyboard input?

``_rebuild_viewports()`` is called whenever the mode changes (Tab press) or the
window is resized. In single-view mode it creates one full-window viewport. In
split-screen mode it creates two side-by-side viewports — left non-interactive
(original camera), right interactive (terrain camera).

During rendering with an FBO (the normal path), the scene is drawn once to a
single full-screen framebuffer at ``render_resolution``. The view-projection
matrix is computed from the active camera mode (original or terrain). Without
an FBO (a fallback path), the renderer iterates the viewport list and draws
the scene once per viewport::

    for viewport in self.viewports:
        self.ctx.viewport = viewport.bounds
        view_proj, view = viewport.get_matrices()
        self._render_scene(view_proj)

Mouse input is routed through the viewports: on click or drag, the renderer
finds which viewport contains the cursor (``_get_viewport_at``), and only
applies camera controls if that viewport is marked ``interactive``. Scroll
events are similarly filtered — scrolling over the non-interactive left pane
is ignored.