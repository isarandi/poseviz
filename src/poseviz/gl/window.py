import glfw
import moderngl


def create_window(
    width: int,
    height: int,
    title: str = "PoseViz",
    vsync: bool = False,
    fullscreen: bool = False,
    samples: int = 4,
) -> tuple:
    """Create GLFW window and ModernGL context.

    Args:
        width, height: Window dimensions (ignored in fullscreen, uses monitor resolution)
        title: Window title
        vsync: Enable vertical sync (caps FPS to monitor refresh rate)
        fullscreen: Enable fullscreen mode
        samples: MSAA samples for antialiasing (0 to disable)

    Returns:
        (window, ctx): GLFW window handle and ModernGL context
    """
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    if samples > 0:
        glfw.window_hint(glfw.SAMPLES, samples)

    monitor = None
    if fullscreen:
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        width, height = mode.size.width, mode.size.height

    window = glfw.create_window(width, height, title, monitor, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if vsync else 0)
    ctx = moderngl.create_context()

    # Enable depth testing
    ctx.enable(moderngl.DEPTH_TEST)

    return window, ctx
