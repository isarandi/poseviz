"""Regenerate the vendored colormap lookup tables.

Samples every colormap registered in matplotlib at 256 points and stores the
resulting uint8 RGB tables in src/poseviz/gl/renderables/colormap_luts.npz,
so that poseviz does not need matplotlib at runtime. Reversed variants
("*_r") are stored explicitly: flipping a sampled table is not bit-identical
to sampling the reversed colormap.

The color data originates from matplotlib (matplotlib license); the viridis
family is CC0 and turbo is Apache-2.0.

Usage: python tools/generate_colormap_luts.py
"""

import pathlib

import matplotlib
import numpy as np

RESOLUTION = 256
OUT_PATH = (
    pathlib.Path(__file__).parent.parent
    / "src" / "poseviz" / "gl" / "renderables" / "colormap_luts.npz"
)


def main():
    x = np.linspace(0, 1, RESOLUTION)
    luts = {}
    for name in sorted(matplotlib.colormaps):
        cmap = matplotlib.colormaps[name]
        luts[name] = (cmap(x)[:, :3] * 255).astype(np.uint8)

    np.savez_compressed(OUT_PATH, **luts)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(luts)} colormaps to {OUT_PATH} ({size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
