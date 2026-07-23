import importlib.resources
import io

import numpy as np
import moderngl

# Lookup tables vendored from matplotlib (see tools/generate_colormap_luts.py);
# loaded lazily on first use.
_luts = None


class Colormap:
    """1D colormap texture for scalar visualization.

    Accepts any matplotlib colormap name (including reversed "*_r" variants);
    the color tables are vendored, so matplotlib is not required at runtime.

    Usage:
        cmap = Colormap.get(ctx, 'viridis')
        cmap.bind(slot=1)
    """

    _cache: dict[str, "Colormap"] = {}
    _ctx: moderngl.Context = None

    def __init__(self, ctx: moderngl.Context, name: str, resolution: int = 256):
        self.name = name
        colors = get_colormap_lut(name, resolution)
        self.texture = ctx.texture((resolution, 1), 3, colors.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        # Clamp to edge: with wrap-around, scalar values at/beyond vmin or vmax
        # would sample a blend of both colormap ends.
        self.texture.repeat_x = False
        self.texture.repeat_y = False

    @classmethod
    def get(cls, ctx: moderngl.Context, name: str = "viridis") -> "Colormap":
        """Get cached colormap for context."""
        if cls._ctx is not ctx:
            for colormap in cls._cache.values():
                try:
                    colormap.texture.release()
                except Exception:
                    pass  # The previous context may already be gone
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


def get_colormap_lut(name: str, resolution: int = 256) -> np.ndarray:
    """Get an RGB lookup table for a colormap name as a (resolution, 3) uint8 array."""
    luts = _load_luts()
    if name not in luts:
        available = ", ".join(sorted(luts))
        raise ValueError(f"Unknown colormap {name!r}. Available: {available}")
    lut = luts[name]
    if resolution != len(lut):
        x = np.linspace(0, 1, resolution)
        xp = np.linspace(0, 1, len(lut))
        lut = np.stack(
            [np.interp(x, xp, lut[:, channel]) for channel in range(3)], axis=-1
        ).astype(np.uint8)
    return np.ascontiguousarray(lut)


def _load_luts() -> dict:
    global _luts
    if _luts is None:
        data = (
            importlib.resources.files("poseviz.gl.renderables")
            .joinpath("colormap_luts.npz")
            .read_bytes()
        )
        _luts = dict(np.load(io.BytesIO(data)))
    return _luts
