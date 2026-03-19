import numba
import numpy as np
import moderngl

from poseviz.gl.renderables.base import ShaderRenderable
from poseviz.gl.renderables.color_source import ColorSource, UniformColor
from poseviz.gl.shader_loader import load_program


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
        self.n_instances = 0
        self.visible = False

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
            self._register_resources(self._wire_program)

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

    def render(self, view_proj: np.ndarray):
        if not self.visible:
            return

        self._set_view_proj(view_proj)
        self.color_source.bind(self.program)
        self.vao.render(moderngl.TRIANGLES)

        if self.wireframe:
            self.ctx.wireframe = True
            self._wire_program["u_view_proj"].write(view_proj.astype(np.float32).tobytes())
            self._wire_program["u_color"].value = (0.0, 0.0, 0.0)
            self._wire_vao.render(moderngl.TRIANGLES)
            self.ctx.wireframe = False

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
