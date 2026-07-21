import numpy as np
import moderngl

from poseviz.gl.renderables.base import ShaderRenderable
from poseviz.gl.renderables.mixins import TextureMixin
from poseviz.gl.transforms import up_basis


class GroundPlaneRenderable(ShaderRenderable, TextureMixin):
    """Renders a checkerboard ground plane."""

    def __init__(
        self, ctx: moderngl.Context, ground_plane_height: float, world_up=(0, -1, 0)
    ):
        super().__init__(ctx)
        self._load_program("textured")

        # Create checkerboard texture
        i, j = np.mgrid[:40, :40]
        pattern = ((i + j) % 2 == 0).astype(np.uint8)
        pattern = np.where(pattern, 255 - 96, 0).astype(np.uint8)
        image = np.stack([pattern, pattern, pattern], axis=-1)
        self._update_texture(image)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Ground plane quad, perpendicular to the world up vector, positioned at
        # ground_plane_height along it
        size = 20000.0
        b1, b2, up = up_basis(world_up)
        center = ground_plane_height * up
        corners = [
            center - size * b1 - size * b2,
            center + size * b1 - size * b2,
            center + size * b1 + size * b2,
            center - size * b1 + size * b2,
        ]
        texcoords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        vertices = np.array(
            [[*corner, *uv] for corner, uv in zip(corners, texcoords)],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        self._register_resources(vbo, ibo)

        self.vao = ctx.vertex_array(
            self.program,
            [(vbo, "3f 2f", "in_position", "in_texcoord")],
            ibo,
        )

    def render(self, view_proj: np.ndarray):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._bind_texture(0)
        self.program["u_opacity"].value = 0.3
        self._set_view_proj(view_proj)

        self.vao.render(moderngl.TRIANGLES)
        self.ctx.disable(moderngl.BLEND)

    def destroy(self):
        self._release_texture()
        super().destroy()
