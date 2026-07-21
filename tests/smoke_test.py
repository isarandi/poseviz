"""Smoke test for the audit fixes: exercises headless runs end to end."""
import os
import pickle
import sys
import time

import numpy as np
import deltacamera
import poseviz

JOINT_NAMES = ["l_wrist", "l_elbow", "l_shoulder", "r_wrist", "r_elbow", "r_shoulder"]
JOINT_EDGES = [[0, 1], [1, 2], [3, 4], [4, 5]]
OUT_DIR = "/var/tmp/poseviz_smoke"
os.makedirs(OUT_DIR, exist_ok=True)


def make_frame(i):
    frame = np.zeros([720, 1280, 3], np.uint8)
    frame[:, :, 0] = int(128 + 127 * np.sin(i / 10))
    return frame


def make_poses(i):
    t = i / 30
    return [np.array(
        [
            [100 * np.cos(t), 100, 2000],
            [50 * np.cos(t), 0, 2000],
            [0, -100, 2000],
            [-100 * np.cos(t), 100, 2000],
            [-50 * np.cos(t), 0, 2000],
            [0, -100, 2000],
        ],
        np.float32,
    )]


def camera(imshape=(720, 1280)):
    return deltacamera.Camera.from_fov(55, imshape)


def test_basic_video_and_trajectory():
    video_path = f"{OUT_DIR}/basic.mp4"
    traj_path = f"{OUT_DIR}/traj.pkl"
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        viz.new_sequence_output(video_path, fps=30, new_camera_trajectory_path=traj_path)
        for i in range(20):
            viz.update(
                frame=make_frame(i),
                boxes=np.array([[100, 100, 200, 200]], np.float32),
                poses=make_poses(i),
                camera=camera(),
            )
        viz.finalize_sequence_output()
    assert os.path.getsize(video_path) > 1000, "video missing/empty"
    with open(traj_path, "rb") as f:
        traj = pickle.load(f)
    assert len(traj) == 20, f"trajectory has {len(traj)} entries, expected 20"
    i_frame, cam = traj[10]
    assert i_frame == 10 and hasattr(cam, "intrinsic_matrix")
    print("basic video + camera trajectory OK")


def test_audio_path_not_overwritten():
    # audio source file must survive (bug C1 used to overwrite it with a pickle)
    audio_path = f"{OUT_DIR}/audio_source.mp4"
    with open(audio_path, "wb") as f:
        f.write(b"FAKE_MP4_CONTENT")
    try:
        with poseviz.PoseViz(
            JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False,
            out_video_path=f"{OUT_DIR}/with_audio.mp4", out_fps=30,
            audio_path=audio_path,
        ) as viz:
            for i in range(5):
                viz.update(frame=make_frame(i), poses=make_poses(i), camera=camera())
            viz.finalize_sequence_output()
    except Exception as e:
        # Muxing fake audio may fail, but the source file must be intact
        print(f"  (note: run failed with {type(e).__name__}: {e})")
    with open(audio_path, "rb") as f:
        assert f.read() == b"FAKE_MP4_CONTENT", "audio source file was overwritten!"
    print("audio source file preserved OK")


def test_pause_resume_close():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        for i in range(3):
            viz.update(frame=make_frame(i), poses=make_poses(i), camera=camera())
        viz.pause()
        for i in range(3):
            viz.update(frame=make_frame(i), poses=make_poses(i), camera=camera())
        time.sleep(0.5)
        viz.resume()
        for i in range(3):
            viz.update(frame=make_frame(i), poses=make_poses(i), camera=camera())
    print("pause/resume/close OK")


def test_start_paused():
    start = time.perf_counter()
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False, paused=True) as viz:
        viz.update(frame=make_frame(0), poses=make_poses(0), camera=camera())
        viz.resume()
        viz.update(frame=make_frame(1), poses=make_poses(1), camera=camera())
    elapsed = time.perf_counter() - start
    assert elapsed < 30, f"paused=True start hung for {elapsed:.0f}s"
    print("start-paused + resume + close OK")


def test_double_close():
    viz = poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False)
    viz.update(frame=make_frame(0), poses=make_poses(0), camera=camera())
    viz.close()
    viz.close()  # must be a no-op, not a hang
    print("double close OK")


def test_missing_camera_raises():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        try:
            viz.update(frame=make_frame(0), poses=make_poses(0))
        except ValueError as e:
            print(f"missing camera raises OK ({e})")
        else:
            raise AssertionError("expected ValueError for frame without camera")


def test_none_and_int_boxes():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        viz.update(frame=make_frame(0), boxes=None, poses=make_poses(0), camera=camera())
        viz.update(
            frame=make_frame(1),
            boxes=np.array([[10, 10, 50, 50]], np.int64),  # integer dtype
            poses=make_poses(1),
            camera=camera(),
        )
    print("None boxes + integer boxes OK")


def test_caller_data_not_mutated():
    from poseviz import ViewInfo
    boxes = np.array([[10.0, 20.0, 100.0, 50.0]], np.float32)
    boxes_orig = boxes.copy()
    vi = ViewInfo(frame=make_frame(0), boxes=boxes, poses=make_poses(0), camera=camera())
    view_infos = (vi,)  # a tuple must be accepted
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        viz.update_multiview(view_infos)
    assert vi.frame is not None, "caller's ViewInfo was mutated (frame replaced)"
    assert np.array_equal(boxes, boxes_orig), "caller's boxes were scaled in place"
    print("caller data not mutated (tuple input) OK")


def test_multiview_growth_and_shrink():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False, n_views=2) as viz:
        from poseviz import ViewInfo
        def vi(i):
            return ViewInfo(
                frame=make_frame(i), boxes=None, poses=make_poses(i), camera=camera()
            )
        viz.update_multiview([vi(0), vi(1)])
        viz.update_multiview([vi(0), vi(1), vi(2)])  # growth beyond ring buffers
        viz.update_multiview([vi(0)])  # shrink
        viz.update_multiview([vi(0), vi(1), vi(2), vi(3)])
    print("multiview growth/shrink OK")


def test_empty_update_raises():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        try:
            viz.update_multiview([])
        except ValueError:
            print("empty update_multiview raises OK")
        else:
            raise AssertionError("expected ValueError for empty view_infos")


def test_world_up_z():
    with poseviz.PoseViz(
        JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False, world_up=(0, 0, 1),
        out_video_path=f"{OUT_DIR}/zup.mp4", out_fps=30,
    ) as viz:
        for i in range(5):
            poses = [p[:, [0, 2, 1]] * np.array([1, 1, -1], np.float32)
                     for p in make_poses(i)]
            viz.update(frame=make_frame(i), poses=poses, camera=camera())
        viz.finalize_sequence_output()
    assert os.path.getsize(f"{OUT_DIR}/zup.mp4") > 1000
    print("world_up=(0,0,1) OK")


def test_uint16_frame():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        frame16 = (make_frame(0).astype(np.uint16)) << 8  # full-range 16-bit
        viz.update(frame=frame16, poses=make_poses(0), camera=camera())
    print("uint16 frame OK")


def test_frameless_view():
    from poseviz import ViewInfo
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        viz.update(frame=make_frame(0), poses=make_poses(0), camera=camera())
        viz.update_multiview(
            [ViewInfo(frame=None, poses=make_poses(1), camera=camera())]
        )
        viz.update(frame=make_frame(2), poses=make_poses(2), camera=camera())
    print("frameless view OK")


def test_confidence_poses():
    with poseviz.PoseViz(JOINT_NAMES, JOINT_EDGES, headless=True, gpu_encode=False) as viz:
        pose4 = np.concatenate(
            [make_poses(0)[0], np.ones((6, 1), np.float32)], axis=1
        )  # (J, 4)
        viz.update(frame=make_frame(0), poses=[pose4], camera=camera())
    print("(J,4) confidence poses OK")


TESTS = [
    test_basic_video_and_trajectory,
    test_audio_path_not_overwritten,
    test_pause_resume_close,
    test_start_paused,
    test_double_close,
    test_missing_camera_raises,
    test_none_and_int_boxes,
    test_caller_data_not_mutated,
    test_multiview_growth_and_shrink,
    test_empty_update_raises,
    test_world_up_z,
    test_uint16_frame,
    test_frameless_view,
    test_confidence_poses,
]

if __name__ == "__main__":
    failed = []
    for test in TESTS:
        print(f"--- {test.__name__}", flush=True)
        try:
            test()
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed.append((test.__name__, repr(e)))

    print("=" * 60)
    if failed:
        print(f"{len(failed)} FAILED: {[name for name, _ in failed]}")
        sys.exit(1)
    print(f"All {len(TESTS)} smoke tests passed.")
