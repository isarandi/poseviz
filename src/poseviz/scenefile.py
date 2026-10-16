"""The .pvz scene file format: a single SQLite file holding a whole visualization scene.

The schema is fully relational (no serialized blobs of structure; numeric
arrays are stored as raw float BLOBs):

- ``scene``: single row of scene-level settings — fps, frame count, world
  conventions, body model spec, mesh-to-world scale.
- ``display``: key-value PoseViz display suggestions for the player
  (appearance only — wireframes, ground plane, image planes. Camera control
  is a player concern: a stored ``viz_camera`` trajectory is a suggestion
  the user can always override interactively; a scene file cannot lock the
  view).
- ``views``: one row per camera view — name, video path, optional static camera.
- ``cameras``: normalized camera storage (intrinsics, extrinsics, distortion
  model and coefficients, image shape), referenced by id from ``views``,
  ``view_cameras`` and ``viz_camera``.
- ``tracks``: per-person shape parameters (betas).
- ``bodies``: SMPL parameters per (frame, track): pose rotvecs and translation.
- ``view_cameras``: time-varying camera trajectories per view.
- ``viz_camera``: optional view-camera trajectory driving the visualization.

The file identifies itself via SQLite's ``application_id`` pragma and carries
the format version in ``user_version``.

Video pixels are not stored; views reference video files by path, resolved
relative to the .pvz file. SMPL parameters are stored in the body model's
native units (meters); ``smpl_to_world_scale`` in the ``scene`` row converts
evaluated meshes to world units (default 1000, i.e. a millimeter world).

Written by :class:`SceneWriter` (pipelines) and read by :class:`SceneReader`
(the ``poseviz`` player CLI, see :mod:`poseviz.player`).
"""

import os
import sqlite3
from contextlib import AbstractContextManager

import deltacamera
import numpy as np

APPLICATION_ID = 0x50565A31  # "PVZ1"
SCENE_FORMAT_VERSION = 1
_COMMIT_EVERY = 1000

_SCHEMA = """
CREATE TABLE scene (
    fps REAL NOT NULL,
    n_frames INTEGER,
    world_up BLOB NOT NULL,             -- 3 float64
    ground_plane_height REAL NOT NULL,
    body_model_name TEXT NOT NULL,
    gender TEXT NOT NULL,
    num_betas INTEGER NOT NULL,
    smpl_to_world_scale REAL NOT NULL);
CREATE TABLE display (key TEXT PRIMARY KEY, value);
CREATE TABLE cameras (
    camera_id INTEGER PRIMARY KEY,
    intrinsics BLOB NOT NULL,           -- 3x3 float64
    rot_world_to_cam BLOB NOT NULL,     -- 3x3 float64
    optical_center BLOB NOT NULL,       -- 3 float64
    world_up BLOB NOT NULL,             -- 3 float64
    distortion_model TEXT,              -- 'BrownConradyEx' | 'FisheyeKannalaBrandt' | NULL
    distortion_coeffs BLOB,             -- float64, length depends on the model
    image_height INTEGER,
    image_width INTEGER);
CREATE TABLE views (
    view INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    video TEXT,
    camera_id INTEGER REFERENCES cameras (camera_id));  -- static camera, NULL if time-varying
CREATE TABLE tracks (track_id INTEGER PRIMARY KEY, betas BLOB NOT NULL);
CREATE TABLE bodies (
    frame INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    pose BLOB NOT NULL,                 -- float32 rotvecs, flat
    trans BLOB NOT NULL,                -- 3 float32
    PRIMARY KEY (frame, track_id));
CREATE TABLE view_cameras (
    view INTEGER NOT NULL,
    frame INTEGER NOT NULL,
    camera_id INTEGER NOT NULL REFERENCES cameras (camera_id),
    PRIMARY KEY (view, frame));
CREATE TABLE viz_camera (
    frame INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras (camera_id));
"""

_DISTORTION_MODELS = {
    'BrownConradyEx': deltacamera.BrownConradyEx,
    'FisheyeKannalaBrandt': deltacamera.FisheyeKannalaBrandt,
}


class SceneWriter(AbstractContextManager):
    """Writes a .pvz scene file incrementally.

    Usage::

        with SceneWriter('out.pvz', fps=30) as sw:
            i_view = sw.add_view(video_path='videos/cam0.mp4', camera=cam0)
            sw.set_track_betas(0, betas)
            for i_frame, (pose, trans) in enumerate(zip(poses, transes)):
                sw.append_body(i_frame, track_id=0, pose_rotvecs=pose, trans=trans)

    The file is usable even if the process dies mid-run: rows are committed
    periodically and the reader infers the frame count when the final one (set
    at close) is missing.
    """

    def __init__(
        self,
        path,
        fps,
        world_up=(0, -1, 0),
        ground_plane_height=-1000.0,
        body_model_name='smpl',
        gender='neutral',
        num_betas=10,
        smpl_to_world_scale=1000.0,
        display=None,
    ):
        """Create a new scene file (an existing file at ``path`` is overwritten).

        Args:
            path: Output path of the .pvz file.
            fps: Frame rate of the scene timeline (and of the referenced videos).
            world_up: Up vector of the world coordinate system.
            ground_plane_height: Ground plane position along the up axis, in world units.
            body_model_name: smplfitter model name ('smpl', 'smplx', ...).
            gender: Body model gender.
            num_betas: Number of shape coefficients stored per track.
            smpl_to_world_scale: Factor from body model units (meters) to world
                units (default 1000 for a millimeter world).
            display: Optional dict of PoseViz constructor defaults for the player
                (e.g. ``{'camera_type': 'original', 'show_field_of_view': False}``).
        """
        if os.path.exists(path):
            os.remove(path)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute(f'PRAGMA application_id = {APPLICATION_ID}')
        self.conn.execute(f'PRAGMA user_version = {SCENE_FORMAT_VERSION}')
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            'INSERT INTO scene (fps, n_frames, world_up, ground_plane_height,'
            ' body_model_name, gender, num_betas, smpl_to_world_scale)'
            ' VALUES (?, NULL, ?, ?, ?, ?, ?, ?)',
            (
                fps,
                np.asarray(world_up, np.float64).tobytes(),
                float(ground_plane_height),
                body_model_name,
                gender,
                int(num_betas),
                float(smpl_to_world_scale),
            ),
        )
        if display:
            self.conn.executemany(
                'INSERT INTO display (key, value) VALUES (?, ?)', list(display.items())
            )
        self._n_appends = 0
        self._max_frame = -1
        self._closed = False
        self.conn.commit()

    def add_view(self, video_path=None, camera=None, name=None):
        """Add a camera view and return its index.

        Args:
            video_path: Path to the video file, preferably relative to the .pvz
                file so the scene folder stays relocatable. None for a view
                without video (camera frustum only).
            camera: Static deltacamera.Camera of this view. Omit for a
                time-varying camera fed via ``append_view_camera``.
            name: Optional display name.
        """
        i_view = self.conn.execute('SELECT COUNT(*) FROM views').fetchone()[0]
        camera_id = self._insert_camera(camera) if camera is not None else None
        self.conn.execute(
            'INSERT INTO views (view, name, video, camera_id) VALUES (?, ?, ?, ?)',
            (
                i_view,
                name if name is not None else f'view{i_view}',
                os.fspath(video_path) if video_path is not None else None,
                camera_id,
            ),
        )
        self.conn.commit()
        return i_view

    def set_track_betas(self, track_id, betas):
        """Set the shape coefficients for a person track."""
        betas = np.asarray(betas, np.float32)
        self.conn.execute(
            'INSERT OR REPLACE INTO tracks (track_id, betas) VALUES (?, ?)',
            (int(track_id), betas.tobytes()),
        )
        self._bump()

    def append_body(self, frame, track_id, pose_rotvecs, trans):
        """Store one person's SMPL parameters for one frame.

        Args:
            frame: Frame index on the scene timeline.
            track_id: Person track id (its betas are set via ``set_track_betas``).
            pose_rotvecs: Flat axis-angle body pose, shape (num_joints*3,).
            trans: Translation in body model units (meters), shape (3,).
        """
        pose = np.asarray(pose_rotvecs, np.float32).reshape(-1)
        trans = np.asarray(trans, np.float32).reshape(3)
        self.conn.execute(
            'INSERT OR REPLACE INTO bodies (frame, track_id, pose, trans) VALUES (?, ?, ?, ?)',
            (int(frame), int(track_id), pose.tobytes(), trans.tobytes()),
        )
        self._max_frame = max(self._max_frame, int(frame))
        self._bump()

    def append_view_camera(self, view, frame, camera):
        """Store the camera of a view for one frame (time-varying cameras)."""
        self.conn.execute(
            'INSERT OR REPLACE INTO view_cameras (view, frame, camera_id) VALUES (?, ?, ?)',
            (int(view), int(frame), self._insert_camera(camera)),
        )
        self._max_frame = max(self._max_frame, int(frame))
        self._bump()

    def append_viz_camera(self, frame, camera):
        """Store the visualization (view) camera for one frame."""
        self.conn.execute(
            'INSERT OR REPLACE INTO viz_camera (frame, camera_id) VALUES (?, ?)',
            (int(frame), self._insert_camera(camera)),
        )
        self._bump()

    def close(self, n_frames=None):
        """Finalize the frame count and close the file.

        Args:
            n_frames: Total frame count of the scene timeline. If None, inferred
                as one past the highest frame index seen.
        """
        if self._closed:
            return
        self._closed = True
        self.conn.execute(
            'UPDATE scene SET n_frames = ?',
            (int(n_frames) if n_frames is not None else self._max_frame + 1,),
        )
        self.conn.commit()
        self.conn.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _insert_camera(self, camera):
        cursor = self.conn.execute(
            'INSERT INTO cameras (intrinsics, rot_world_to_cam, optical_center, world_up,'
            ' distortion_model, distortion_coeffs, image_height, image_width)'
            ' VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            camera_to_row(camera),
        )
        return cursor.lastrowid

    def _bump(self):
        self._n_appends += 1
        if self._n_appends % _COMMIT_EVERY == 0:
            self.conn.commit()


class SceneReader:
    """Reads a .pvz scene file, loading all per-frame data into memory.

    Parametric scene data is small (hundreds of bytes per person per frame), so
    everything except video pixels is held in RAM, giving random access to any
    frame. Scene-level settings are exposed as attributes: ``fps``,
    ``n_frames``, ``world_up``, ``ground_plane_height``, ``body_model`` (dict
    with model_name/gender/num_betas), ``smpl_to_world_scale``, ``display``
    (dict), ``views`` (list of dicts with name/video/camera), ``betas``
    (track_id -> array).
    """

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.path = path
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            app_id = conn.execute('PRAGMA application_id').fetchone()[0]
            if app_id != APPLICATION_ID:
                raise ValueError(f'{path} is not a .pvz scene file.')
            version = conn.execute('PRAGMA user_version').fetchone()[0]
            if version != SCENE_FORMAT_VERSION:
                raise ValueError(
                    f'{path}: scene format version {version} is not supported '
                    f'(this reader supports version {SCENE_FORMAT_VERSION}).'
                )

            row = conn.execute(
                'SELECT fps, n_frames, world_up, ground_plane_height,'
                ' body_model_name, gender, num_betas, smpl_to_world_scale FROM scene'
            ).fetchone()
            if row is None:
                raise ValueError(f'{path}: empty scene table.')
            (fps, n_frames, world_up, ground_plane_height, model_name, gender, num_betas,
             scale) = row
            self.fps = fps
            self.world_up = tuple(np.frombuffer(world_up, np.float64))
            self.ground_plane_height = ground_plane_height
            self.body_model = dict(model_name=model_name, gender=gender, num_betas=num_betas)
            self.smpl_to_world_scale = scale

            self.display = dict(conn.execute('SELECT key, value FROM display'))

            cameras = {
                row[0]: camera_from_row(row[1:])
                for row in conn.execute(
                    'SELECT camera_id, intrinsics, rot_world_to_cam, optical_center,'
                    ' world_up, distortion_model, distortion_coeffs, image_height,'
                    ' image_width FROM cameras'
                )
            }

            self.views = [
                dict(name=name, video=video, camera=cameras.get(camera_id))
                for name, video, camera_id in conn.execute(
                    'SELECT name, video, camera_id FROM views ORDER BY view'
                )
            ]

            self.betas = {
                track_id: np.frombuffer(blob, np.float32)
                for track_id, blob in conn.execute('SELECT track_id, betas FROM tracks')
            }
            self._bodies = {}  # frame -> list of (track_id, pose, trans)
            for frame, track_id, pose, trans in conn.execute(
                'SELECT frame, track_id, pose, trans FROM bodies ORDER BY frame, track_id'
            ):
                self._bodies.setdefault(frame, []).append(
                    (track_id, np.frombuffer(pose, np.float32), np.frombuffer(trans, np.float32))
                )
            self._view_cameras = {
                (view, frame): cameras[camera_id]
                for view, frame, camera_id in conn.execute(
                    'SELECT view, frame, camera_id FROM view_cameras'
                )
            }
            self._viz_cameras = {
                frame: cameras[camera_id]
                for frame, camera_id in conn.execute('SELECT frame, camera_id FROM viz_camera')
            }
        finally:
            conn.close()

        if n_frames is None:
            # The writer died before close(); infer from the stored data.
            n_frames = 1 + max(
                [f for f in self._bodies]
                + [f for _, f in self._view_cameras]
                + [-1]
            )
        self.n_frames = n_frames

    def bodies(self, frame):
        """The people present in a frame, as a list of (track_id, pose_rotvecs, trans)."""
        return self._bodies.get(frame, [])

    def view_camera(self, i_view, frame):
        """The camera of a view at a frame: static if set, else from the trajectory."""
        static = self.views[i_view]['camera']
        if static is not None:
            return static
        return self._view_cameras.get((i_view, frame))

    def viz_camera(self, frame):
        """The stored visualization camera for a frame, or None."""
        return self._viz_cameras.get(frame)

    @property
    def has_viz_camera(self):
        return bool(self._viz_cameras)

    @property
    def n_person_frames(self):
        return sum(len(v) for v in self._bodies.values())

    def resolve_path(self, p):
        """Resolve a (video) path from the manifest relative to the .pvz file."""
        if p is None or os.path.isabs(p):
            return p
        return os.path.join(os.path.dirname(os.path.abspath(self.path)), p)


def camera_to_row(camera):
    """Serialize a deltacamera.Camera to a row for the ``cameras`` table."""
    dist_model = camera.distortion_model
    if dist_model is not None:
        model_name = type(dist_model).__name__
        if model_name not in _DISTORTION_MODELS:
            raise ValueError(f'Cannot serialize distortion model {model_name}.')
        dist_coeffs = np.asarray(dist_model.coeffs, np.float64).tobytes()
    else:
        model_name = None
        dist_coeffs = None
    image_shape = camera.image_shape
    return (
        np.asarray(camera.intrinsic_matrix, np.float64).tobytes(),
        np.asarray(camera.R, np.float64).tobytes(),
        np.asarray(camera.t, np.float64).tobytes(),
        np.asarray(camera.world_up, np.float64).tobytes(),
        model_name,
        dist_coeffs,
        int(image_shape[0]) if image_shape is not None else None,
        int(image_shape[1]) if image_shape is not None else None,
    )


def camera_from_row(row):
    """Deserialize a ``cameras`` table row (without camera_id) to a Camera."""
    (intrinsics, rot, optical_center, world_up, model_name, dist_coeffs, image_h,
     image_w) = row
    if model_name is not None:
        dist_model = _DISTORTION_MODELS[model_name](np.frombuffer(dist_coeffs, np.float64))
    else:
        dist_model = None
    return deltacamera.Camera(
        intrinsic_matrix=np.frombuffer(intrinsics, np.float64).reshape(3, 3).astype(np.float32),
        rot_world_to_cam=np.frombuffer(rot, np.float64).reshape(3, 3).astype(np.float32),
        optical_center=np.frombuffer(optical_center, np.float64).astype(np.float32),
        distortion_model=dist_model,
        world_up=np.frombuffer(world_up, np.float64).astype(np.float32),
        image_shape=(image_h, image_w) if image_h is not None else None,
    )
