"""Tests for the .pvz scene file format and the player CLI."""
import os
import subprocess
import sys

import numpy as np
import deltacamera

from poseviz.scenefile import SceneReader, SceneWriter

OUT_DIR = "/var/tmp/poseviz_scene_test"
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES = 30
IMSHAPE = (240, 320)
BODY_MODELS_AVAILABLE = os.path.isdir("/work/sarandi/data/body_models/smpl")


def make_camera(i=0):
    cam = deltacamera.Camera.from_fov(55, IMSHAPE, world_up=(0, -1, 0))
    return cam.orbited_around(np.array([0.0, 0.0, 3000.0]), np.deg2rad(i), "horizontal")


def make_pose(i, phase=0.0):
    pose = np.zeros(72, np.float32)
    pose[50] = 0.5 * np.sin(i / 10 + phase)  # wiggle one arm joint
    return pose


def write_scene(path, with_videos=True):
    video_paths = []
    if with_videos:
        import framepump

        for i_view in range(2):
            vp = f"video{i_view}.mp4"
            with framepump.VideoWriter(os.path.join(OUT_DIR, vp), fps=30) as vw:
                for i in range(N_FRAMES):
                    frame = np.full([*IMSHAPE, 3], 30 * i_view + 3 * i, np.uint8)
                    vw.append_data(frame)
            video_paths.append(vp)
    else:
        video_paths = [None, None]

    with SceneWriter(
        path, fps=30, world_up=(0, -1, 0), display={"show_field_of_view": False}
    ) as sw:
        sw.add_view(video_path=video_paths[0], camera=make_camera(), name="static_cam")
        i_view1 = sw.add_view(video_path=video_paths[1], name="moving_cam")
        for track_id, phase in [(0, 0.0), (1, 2.0)]:
            sw.set_track_betas(track_id, np.full(10, 0.1 * track_id, np.float32))
        for i in range(N_FRAMES):
            for track_id, phase in [(0, 0.0), (1, 2.0)]:
                trans = np.array([0.6 * track_id - 0.3, 0.9, 3.0], np.float32)
                trans[0] += 0.2 * np.sin(i / 15 + phase)
                sw.append_body(i, track_id, make_pose(i, phase), trans)
            sw.append_view_camera(i_view1, i, make_camera(i))
            sw.append_viz_camera(i, make_camera(-i / 2))
    return path


def test_roundtrip():
    path = write_scene(os.path.join(OUT_DIR, "roundtrip.pvz"), with_videos=False)
    reader = SceneReader(path)

    assert reader.fps == 30 and reader.n_frames == N_FRAMES
    assert reader.display == {"show_field_of_view": False}
    assert tuple(reader.world_up) == (0, -1, 0)
    assert reader.body_model == {"model_name": "smpl", "gender": "neutral", "num_betas": 10}
    assert len(reader.views) == 2
    assert reader.views[0]["name"] == "static_cam"
    assert reader.views[0]["camera"] is not None and reader.views[1]["camera"] is None

    assert set(reader.betas.keys()) == {0, 1}
    assert np.array_equal(reader.betas[1], np.full(10, 0.1, np.float32))

    bodies = reader.bodies(7)
    assert [b[0] for b in bodies] == [0, 1]
    assert np.array_equal(bodies[0][1], make_pose(7, 0.0))
    assert bodies[1][2].shape == (3,)

    cam_static = reader.view_camera(0, 12)
    assert cam_static.allclose(make_camera())
    cam_moving = reader.view_camera(1, 12)
    assert cam_moving.allclose(make_camera(12))
    assert reader.has_viz_camera and reader.viz_camera(3) is not None
    assert reader.viz_camera(N_FRAMES + 5) is None
    assert reader.bodies(N_FRAMES + 5) == []

    assert reader.resolve_path("x.mp4") == os.path.join(OUT_DIR, "x.mp4")
    assert reader.resolve_path("/abs/x.mp4") == "/abs/x.mp4"
    print("scene roundtrip OK")


def test_not_a_scene_file_rejected():
    import sqlite3

    path = os.path.join(OUT_DIR, "not_a_scene.pvz")
    if os.path.exists(path):
        os.remove(path)
    sqlite3.connect(path).close()  # a valid but foreign SQLite file
    try:
        SceneReader(path)
    except ValueError as e:
        print(f"foreign SQLite file rejected OK ({e})")
    else:
        raise AssertionError("expected ValueError for a non-.pvz SQLite file")


def test_distortion_camera_roundtrip():
    from poseviz.scenefile import camera_to_row, camera_from_row

    intr = np.array([[1000.0, 0, 160], [0, 1000.0, 120], [0, 0, 1]], np.float32)
    for dist_model in [
        deltacamera.BrownConradyEx(np.array([-0.2, 0.05, 0.001, 0.0005, -0.01])),
        deltacamera.FisheyeKannalaBrandt(np.array([0.1, 0.02, 0.003, 0.0004])),
        None,
    ]:
        cam = deltacamera.Camera(
            intrinsic_matrix=intr,
            rot_world_to_cam=np.eye(3, dtype=np.float32),
            optical_center=np.array([10.0, 20.0, -500.0], np.float32),
            distortion_model=dist_model,
            world_up=(0, -1, 0),
            image_shape=IMSHAPE,
        )
        cam2 = camera_from_row(camera_to_row(cam))
        assert cam2.allclose(cam)
        assert cam2.has_distortion() == (dist_model is not None)
        assert cam2.has_fisheye_distortion() == isinstance(
            dist_model, deltacamera.FisheyeKannalaBrandt
        )
    print("distortion camera roundtrip OK (BrownConradyEx, fisheye, none)")


def test_info_cli():
    path = write_scene(os.path.join(OUT_DIR, "info.pvz"), with_videos=False)
    r = subprocess.run(
        [sys.executable, "-m", "poseviz.player", path, "--info"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert "2 frames" not in r.stdout and f"{N_FRAMES} frames" in r.stdout
    assert "static_cam" in r.stdout and "moving_cam" in r.stdout
    print("--info CLI OK")


def test_player_end_to_end():
    if not BODY_MODELS_AVAILABLE:
        print("SKIP player end-to-end (body model files not available)")
        return
    path = write_scene(os.path.join(OUT_DIR, "scene.pvz"), with_videos=True)
    out_video = os.path.join(OUT_DIR, "playback.mp4")
    if os.path.exists(out_video):
        os.remove(out_video)
    env = dict(os.environ, DATA_ROOT="/work/sarandi/data")
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    r = subprocess.run(
        [sys.executable, "-m", "poseviz.player", path,
         "--out-video", out_video, "--skeleton", "--speed", "100"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert os.path.getsize(out_video) > 1000, "playback video missing/empty"

    # The playback must actually show something (bodies, ground plane), not a blank canvas
    import framepump

    frames = list(framepump.VideoFrames(out_video))
    assert len(frames) == N_FRAMES, f"expected {N_FRAMES} playback frames, got {len(frames)}"
    mid = frames[len(frames) // 2]
    white_fraction = (mid > 250).all(axis=-1).mean()
    assert white_fraction < 0.95, f"playback looks blank (white fraction {white_fraction:.3f})"
    print("player end-to-end OK")


TESTS = [
    test_roundtrip,
    test_not_a_scene_file_rejected,
    test_distortion_camera_roundtrip,
    test_info_cli,
    test_player_end_to_end,
]

if __name__ == "__main__":
    failed = []
    for test in TESTS:
        print(f"--- {test.__name__}", flush=True)
        try:
            test()
        except Exception:
            import traceback

            traceback.print_exc()
            failed.append(test.__name__)
    print("=" * 60)
    if failed:
        print(f"{len(failed)} FAILED: {failed}")
        sys.exit(1)
    print(f"All {len(TESTS)} scene tests passed.")
