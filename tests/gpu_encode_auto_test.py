"""Headless video output with the default gpu_encode=None must work whether or not the
zero-copy NVENC path is available: on a machine (or environment) without NVIDIA EGL or
without framepump's CUDA bindings, PoseViz falls back to reading the frames back and
writing them with framepump.VideoWriter. gpu_encode=True must instead fail with a
VisualizerError (not hang) when the zero-copy path cannot work.

Both paths must also write the video upright (they differ in how the rendered image
reaches the encoder).

Run: python tests/gpu_encode_auto_test.py
Run it in an environment with and in one without ``framepump[nvenc-cuda]`` to cover
both branches; each scenario runs in a subprocess under a timeout.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

JOINT_NAMES = ["l_wrist", "l_elbow", "l_shoulder", "r_wrist", "r_elbow", "r_shoulder"]
JOINT_EDGES = [[0, 1], [1, 2], [3, 4], [4, 5]]
N_FRAMES = 20


def main():
    from framepump.video_writing_gl import available_backends

    backends = available_backends()
    print(f"framepump GL backends here: {backends}")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "auto.mp4")
        res = run_scenario("auto", out)
        assert res.returncode == 0, res.stdout + res.stderr
        n = count_frames(out)
        assert n == N_FRAMES, f"expected {N_FRAMES} frames, got {n}"
        top, bottom = red_of_halves(out)
        assert top > bottom + 100, f"video upside down: red top {top:.0f}, bottom {bottom:.0f}"
        # Without the CUDA backend the zero-copy path cannot run headless (the strict
        # scenario below proves it), so a complete video here came from the fallback.
        path = "zero-copy NVENC" if "cuda" in backends else "read-back fallback"
        print(f"gpu_encode=None: wrote {n} frames ({path})")

        if "cuda" not in backends:
            res = run_scenario("strict", os.path.join(tmp, "strict.mp4"))
            assert res.returncode == 0, res.stdout + res.stderr
            print("gpu_encode=True without the CUDA backend: VisualizerError raised")
    print("OK")


def run_scenario(mode, out):
    return subprocess.run(
        [sys.executable, __file__, mode, out], capture_output=True, text=True, timeout=120,
        env={k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")},
    )


def scenario(mode, out):
    import logging

    import deltacamera
    import poseviz
    from poseviz.errors import VisualizerError

    logging.basicConfig(level=logging.INFO)
    cam = deltacamera.Camera.from_fov(55, (480, 640))
    poses = [np.array([[0, 100, 2000], [50, 0, 2000], [0, -100, 2000],
                       [-100, 100, 2000], [-50, 0, 2000], [0, -100, 2000]], np.float32)]
    frame = np.zeros([480, 640, 3], np.uint8)
    frame[:240] = (255, 0, 0)  # red top half: the recorded video must keep it on top
    gpu_encode = None if mode == "auto" else True
    viz = poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=gpu_encode)
    try:
        viz.new_sequence_output(out, fps=30)
        for _ in range(N_FRAMES):
            viz.update(frame=frame, boxes=np.zeros((0, 4), np.float32), poses=poses, camera=cam)
        viz.finalize_sequence_output()
        viz.close()
    except VisualizerError:
        if mode == "strict":
            viz.close(raise_on_error=False)
            return
        raise
    if mode == "strict":
        raise AssertionError("gpu_encode=True should have failed without the CUDA backend")


def red_of_halves(path):
    """Mean red of the image's top and bottom thirds in the last frame of the video."""
    import av

    with av.open(path) as c:
        f = list(c.decode(video=0))[-1].to_ndarray(format="rgb24")
    h = f.shape[0]
    return f[: h // 3, :, 0].mean(), f[-h // 3 :, :, 0].mean()


def count_frames(path):
    import av

    with av.open(path) as c:
        return sum(1 for _ in c.decode(video=0))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        scenario(sys.argv[1], sys.argv[2])
    else:
        main()
