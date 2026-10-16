"""When the visualizer process dies mid-run (e.g. an encoder OOM; here:
SIGKILL), the parent's update must raise a VisualizerError instead of
blocking, and the parent process must EXIT cleanly even though close() is
never called (the mp.Queue feeder join at interpreter exit must not deadlock
on a pipe the dead child can no longer drain).

Run: python tests/dead_child_test.py            (child mode, exits 0 on success)
     the __main__ block re-launches itself under a timeout as the actual test.
"""
import os
import subprocess
import sys
import time

import numpy as np

JOINT_NAMES = ["l_wrist", "l_elbow", "l_shoulder", "r_wrist", "r_elbow", "r_shoulder"]
JOINT_EDGES = [[0, 1], [1, 2], [3, 4], [4, 5]]


def scenario():
    import deltacamera
    import poseviz
    from poseviz.errors import VisualizerError

    cam = deltacamera.Camera.from_fov(55, (720, 1280))
    poses = [np.array([[0, 100, 2000], [50, 0, 2000], [0, -100, 2000],
                       [-100, 100, 2000], [-50, 0, 2000], [0, -100, 2000]], np.float32)]
    viz = poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, queue_size=4)
    frame = np.zeros([720, 1280, 3], np.uint8)
    for i in range(5):
        viz.update(frame=frame, boxes=np.zeros((0, 4), np.float32), poses=poses, camera=cam)
    # simulate the child crashing hard (NVENC OOM class)
    viz.visualizer_process.kill()
    t0 = time.monotonic()
    try:
        for i in range(100):
            viz.update(frame=frame, boxes=np.zeros((0, 4), np.float32), poses=poses, camera=cam)
            time.sleep(0.02)
    except VisualizerError as e:
        print(f'OK: VisualizerError after {time.monotonic()-t0:.1f}s: {str(e)[:80]}', flush=True)
    else:
        print('FAIL: no VisualizerError after child death', flush=True)
        os._exit(3)
    if time.monotonic() - t0 > 30:
        print('FAIL: detection took too long', flush=True)
        os._exit(4)
    print('exiting WITHOUT close(); interpreter exit must not hang', flush=True)
    # deliberate: no viz.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'child':
        scenario()
        sys.exit(0)
    r = subprocess.run([sys.executable, os.path.abspath(__file__), 'child'],
                       timeout=120, capture_output=True, text=True)
    sys.stdout.write(r.stdout[-2000:])
    sys.stderr.write(r.stderr[-1500:])
    ok = r.returncode == 0 and 'OK: VisualizerError' in r.stdout
    print('DEAD-CHILD TEST', 'PASSED' if ok else f'FAILED (rc={r.returncode})')
    sys.exit(0 if ok else 1)
