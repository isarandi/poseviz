import importlib.resources
import moderngl


def load_program(ctx: moderngl.Context, name: str) -> moderngl.Program:
    """Load a shader program by name (e.g., 'sphere' loads sphere.vert and sphere.frag)."""
    vert_src = _load_shader_source(f"{name}.vert")
    frag_src = _load_shader_source(f"{name}.frag")

    # Check for optional geometry shader
    try:
        geom_src = _load_shader_source(f"{name}.geom")
    except FileNotFoundError:
        geom_src = None

    return ctx.program(
        vertex_shader=vert_src,
        fragment_shader=frag_src,
        geometry_shader=geom_src,
    )


def _load_shader_source(filename: str) -> str:
    """Load shader source from the shaders directory."""
    ref = importlib.resources.files("poseviz.gl.shaders").joinpath(filename)
    return ref.read_text()
