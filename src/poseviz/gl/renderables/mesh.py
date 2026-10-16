import logging

import numba
import numpy as np
import moderngl

from poseviz.gl.renderables.base import ShaderRenderable
from poseviz.gl.renderables.color_source import ColorSource, UniformColor
from poseviz.gl.shader_loader import load_program

logger = logging.getLogger(__name__)


class MeshRenderable(ShaderRenderable):
    """Renders triangle meshes with dynamic vertex updates.

    Supports multiple instances of the same mesh topology (e.g., multiple SMPL bodies).
    Pass a list of vertex arrays to update() for multi-mesh rendering.

    Color is determined by the ColorSource strategy:
    - UniformColor: single color for entire mesh
    - VertexRGBColor: per-vertex RGB
    - ScalarColormapColor: per-vertex scalar with colormap
    - TextureColor: UV-mapped texture

    Example:
        # Uniform color
        renderer = MeshRenderable(ctx, faces, UniformColor((1.0, 0.5, 0.0)))

        # Per-vertex scalar with colormap
        renderer = MeshRenderable(ctx, faces, ScalarColormapColor('viridis', vmin=0, vmax=1))
        renderer.update(vertices, color_data=scalars)

        # Multiple meshes
        renderer.update([vertices1, vertices2])
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        faces: np.ndarray,
        color_source: ColorSource = None,
        max_vertices: int = 100000,
        max_instances: int = 10,
        wireframe: bool = True,
    ):
        super().__init__(ctx)

        if color_source is None:
            color_source = UniformColor((0.7, 0.7, 0.7))

        self.color_source = color_source
        self.wireframe = wireframe
        self.base_faces = faces.astype(np.uint32)
        self.n_verts_per_mesh = None  # Set on first update
        self.max_vertices = max_vertices
        self.max_instances = max_instances
        self.vertex_count = 0
        self.index_count = 0  # Number of indices currently valid in the IBO
        self.n_instances = 0
        self.visible = False
        self._warned_clamp = False
        # Current geometry on the CPU side, kept for the per-frame back-to-front
        # face sort of translucent meshes; center_world serves the renderer's
        # object-level sort of translucent renderables.
        self._flat_verts = None
        self._combined_faces = None
        self.center_world = None

        # Load shader based on color source
        self._load_program(color_source.shader_name)

        # Set up color source GPU resources
        color_source.setup(ctx, max_vertices)

        # Pre-allocate vertex buffers
        vbo_positions = ctx.buffer(reserve=max_vertices * 12)
        vbo_normals = ctx.buffer(reserve=max_vertices * 12)

        # IBO sized for max_instances meshes
        max_faces = len(self.base_faces) * max_instances
        self._ibo = ctx.buffer(reserve=max_faces * 12)  # 3 uint32 per face
        self._register_resources(vbo_positions, vbo_normals, self._ibo)
        self._vbo_positions = vbo_positions
        self._vbo_normals = vbo_normals

        # Build VAO with geometry + color source attributes
        vao_content = [
            (vbo_positions, "3f", "in_position"),
            (vbo_normals, "3f", "in_normal"),
        ] + color_source.get_vao_content()

        self.vao = ctx.vertex_array(self.program, vao_content, self._ibo)

        if wireframe:
            self._wire_program = load_program(ctx, "line")
            self._wire_vao = ctx.vertex_array(
                self._wire_program, [(vbo_positions, "3f", "in_position")], self._ibo
            )
            self._register_resources(self._wire_program, self._wire_vao)

        # Single-pass "solid wireframe" program for translucent uniform-colour
        # meshes: faces and edges of every triangle in one sorted draw call
        # (see render). Other colour sources fall back to the layered draw.
        self._solid_wire_vao = None
        if wireframe and color_source.shader_name == "mesh":
            self._solid_wire_program = load_program(ctx, "mesh_wire")
            self._solid_wire_vao = ctx.vertex_array(
                self._solid_wire_program, vao_content, self._ibo
            )
            self._register_resources(self._solid_wire_program, self._solid_wire_vao)

    def update(self, vertices, color_data=None):
        """Update mesh vertices and optionally color data.

        Args:
            vertices: (N, 3) array for single mesh, or (M, N, 3) / list of (N, 3) for M meshes
            color_data: Data for color source (format depends on ColorSource type):
                - UniformColor: tuple (r, g, b)
                - VertexRGBColor: (N, 3) array
                - ScalarColormapColor: (N,) array of scalars
                - TextureColor: image array (set texcoords separately via set_texcoords)
        """
        # Handle list of vertex arrays (multiple meshes)
        if isinstance(vertices, list):
            if len(vertices) == 0:
                self.visible = False
                return
            vertices = np.stack(vertices, axis=0)  # (M, N, 3)

        vertices = np.asarray(vertices, dtype=np.float32)

        # Single mesh: (N, 3), multiple meshes: (M, N, 3)
        if vertices.ndim == 2:
            vertices = vertices[np.newaxis]  # (1, N, 3)

        n_instances, n_verts, _ = vertices.shape

        if n_instances == 0 or n_verts == 0:
            self.visible = False
            return

        # Clamp to the preallocated buffer sizes: an oversized write would
        # raise a GL error and kill the visualizer process.
        n_fit = min(self.max_instances, self.max_vertices // n_verts)
        if n_instances > n_fit:
            if not self._warned_clamp:
                logger.warning(
                    f"Too many mesh instances ({n_instances}); only {n_fit} are "
                    f"shown. Increase max_instances/max_vertices of MeshRenderable."
                )
                self._warned_clamp = True
            if n_fit == 0:
                self.visible = False
                return
            vertices = vertices[:n_fit]
            n_instances = n_fit

        # Update faces if instance count changed
        if n_instances != self.n_instances or self.n_verts_per_mesh != n_verts:
            self._update_faces(n_instances, n_verts)

        # Flatten vertices: (M, N, 3) -> (M*N, 3)
        flat_verts = vertices.reshape(-1, 3)

        # Compute normals for combined mesh
        combined_faces = self._get_combined_faces(n_instances, n_verts)
        normals = _compute_normals(flat_verts, combined_faces)

        self._vbo_positions.write(flat_verts.tobytes())
        self._vbo_normals.write(normals.tobytes())
        self.vertex_count = len(flat_verts)
        self.visible = True
        self._flat_verts = flat_verts
        self._combined_faces = combined_faces
        self.center_world = flat_verts.astype(np.float64).mean(0)

        if color_data is not None:
            self.color_source.update(color_data)

    def _get_combined_faces(self, n_instances: int, n_verts: int) -> np.ndarray:
        """Get face indices for multiple mesh instances."""
        if n_instances == 1:
            return self.base_faces

        # Replicate faces with offset for each instance
        all_faces = []
        for i in range(n_instances):
            offset = i * n_verts
            all_faces.append(self.base_faces + offset)
        return np.concatenate(all_faces, axis=0)

    def _update_faces(self, n_instances: int, n_verts: int):
        """Update IBO with replicated faces for multiple instances."""
        self.n_instances = n_instances
        self.n_verts_per_mesh = n_verts
        combined_faces = self._get_combined_faces(n_instances, n_verts)
        self._ibo.write(combined_faces.astype(np.uint32).tobytes())
        self.index_count = combined_faces.size

    def render(self, view_proj: np.ndarray, eye=None):
        """Draw the mesh.

        A translucent mesh (color source with ``needs_blend``) is alpha-blended
        without depth writes, and when ``eye`` (viewer position in world
        coordinates) is given its faces are first sorted back to front for
        this viewpoint: blending is order-dependent, and a closed body covers
        every pixel with at least two layers (back and front skin), so the
        static index order would composite them wrongly. For a uniform-colour
        mesh the edges are then drawn by the fragment shader (mesh_wire
        shaders), so faces and edges of both skins go through that one sorted
        draw call. Other colour sources are drawn as two layers instead, far
        skin first (back-facing triangles with their edges, then front-facing
        ones with theirs), which is exact only where no near part's back
        skin lies in front of a far part's front skin.
        """
        if not self.visible or self.index_count == 0:
            return

        needs_blend = getattr(self.color_source, "needs_blend", False)
        if not needs_blend:
            self._draw_faces(view_proj)
            if self.wireframe:
                self._draw_wire(view_proj, 1.0)
            return

        if eye is not None:
            self._sort_faces_back_to_front(eye)
        opacity = float(getattr(self.color_source, "opacity", 1.0))
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.fbo.depth_mask = False
        if self._solid_wire_vao is not None:
            self._draw_solid_wire(view_proj, opacity)
        elif not self.wireframe:
            self._draw_faces(view_proj)
        else:
            self.ctx.enable(moderngl.CULL_FACE)
            for cull in ("front", "back"):  # culling 'front' leaves the far skin
                self.ctx.cull_face = cull
                self._draw_faces(view_proj)
                self._draw_wire(view_proj, opacity)
            self.ctx.cull_face = "back"
            self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.fbo.depth_mask = True
        self.ctx.disable(moderngl.BLEND)

    def _draw_solid_wire(self, view_proj, opacity):
        prog = self._solid_wire_program
        prog["u_view_proj"].write(view_proj.astype(np.float32).tobytes())
        if self._current_view is not None:
            prog["u_view"].write(self._current_view)
        prog["u_color"].value = self.color_source.color
        prog["u_opacity"].value = opacity
        prog["u_edge_color"].value = (0.0, 0.0, 0.0)
        prog["u_edge_width"].value = 1.0
        self._solid_wire_vao.render(moderngl.TRIANGLES, vertices=self.index_count)

    def _draw_faces(self, view_proj):
        self._set_view_proj(view_proj)
        self.color_source.bind(self.program)
        # Pass the valid index count explicitly: the IBO is reserved for
        # max_instances meshes and its unwritten tail must not be drawn.
        self.vao.render(moderngl.TRIANGLES, vertices=self.index_count)

    def _draw_wire(self, view_proj, opacity):
        self.ctx.wireframe = True
        self._wire_program["u_view_proj"].write(view_proj.astype(np.float32).tobytes())
        self._wire_program["u_color"].value = (0.0, 0.0, 0.0)
        self._wire_program["u_opacity"].value = opacity
        self._wire_vao.render(moderngl.TRIANGLES, vertices=self.index_count)
        self.ctx.wireframe = False

    def _sort_faces_back_to_front(self, eye):
        """Rewrite the IBO with the faces of all instances ordered by decreasing
        distance of their centroid from ``eye`` (painter's order). About 1 ms
        for a SMPL body; exact except for mutually intersecting faces."""
        faces = self._combined_faces
        centroids = self._flat_verts[faces].mean(1)
        d2 = ((centroids - np.asarray(eye, np.float32)) ** 2).sum(1)
        order = np.argsort(-d2, kind="stable")
        self._ibo.write(np.ascontiguousarray(faces[order], dtype=np.uint32).tobytes())

    def destroy(self):
        self.color_source.release()
        super().destroy()


@numba.njit(error_model='numpy', cache=True)
def _compute_normals(vertices, faces):
    """Compute per-vertex normals from faces.

    Single pass over faces: compute face normal inline and scatter-add to vertices.
    Then normalize. All in one tight loop — no intermediate arrays.
    """
    n_verts = vertices.shape[0]
    n_faces = faces.shape[0]
    normals = np.zeros((n_verts, 3), np.float32)

    for i in range(n_faces):
        i0 = faces[i, 0]
        i1 = faces[i, 1]
        i2 = faces[i, 2]

        # Edge vectors
        e1x = vertices[i1, 0] - vertices[i0, 0]
        e1y = vertices[i1, 1] - vertices[i0, 1]
        e1z = vertices[i1, 2] - vertices[i0, 2]
        e2x = vertices[i2, 0] - vertices[i0, 0]
        e2y = vertices[i2, 1] - vertices[i0, 1]
        e2z = vertices[i2, 2] - vertices[i0, 2]

        # Cross product (face normal, unnormalized — area-weighted)
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        # Scatter-add to each vertex of this face
        normals[i0, 0] += nx
        normals[i0, 1] += ny
        normals[i0, 2] += nz
        normals[i1, 0] += nx
        normals[i1, 1] += ny
        normals[i1, 2] += nz
        normals[i2, 0] += nx
        normals[i2, 1] += ny
        normals[i2, 2] += nz

    # Normalize
    for i in range(n_verts):
        nx = normals[i, 0]
        ny = normals[i, 1]
        nz = normals[i, 2]
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        if length > np.float32(1e-8):
            normals[i, 0] /= length
            normals[i, 1] /= length
            normals[i, 2] /= length

    return normals
