import numpy as np
import moderngl
import matplotlib.pyplot as plt


class Colormap:
    """1D colormap texture for scalar visualization.

    Usage:
        cmap = Colormap.get(ctx, 'viridis')
        cmap.bind(slot=1)
    """

    _cache: dict[str, "Colormap"] = {}
    _ctx: moderngl.Context = None

    def __init__(self, ctx: moderngl.Context, name: str, resolution: int = 256):
        self.name = name
        cmap = plt.get_cmap(name)
        colors = (cmap(np.linspace(0, 1, resolution))[:, :3] * 255).astype(np.uint8)
        self.texture = ctx.texture((resolution, 1), 3, colors.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    @classmethod
    def get(cls, ctx: moderngl.Context, name: str = "viridis") -> "Colormap":
        """Get cached colormap for context."""
        if cls._ctx is not ctx:
            cls._cache.clear()
            cls._ctx = ctx

        if name not in cls._cache:
            cls._cache[name] = cls(ctx, name)
        return cls._cache[name]

    def bind(self, slot: int = 1):
        """Bind colormap texture to slot."""
        self.texture.use(slot)

    def release(self):
        """Release texture. Usually not needed (cache manages lifetime)."""
        self.texture.release()
