"""Command-line player for .pvz scene files (see :mod:`poseviz.scenefile`).

Usage::

    poseviz scene.pvz               # play interactively (or headless without display)
    poseviz scene.pvz --info        # print a summary and exit
    poseviz scene.pvz --out-video out.mp4

The player is an ordinary PoseViz producer: it decodes the referenced videos
with framepump, evaluates the stored SMPL parameters with smplfitter and feeds
``update_multiview``. All interactive controls of PoseViz apply, including
taking over the camera from a stored visualization-camera trajectory by
dragging with the mouse.

smplfitter is required only by this player (not by the poseviz package
itself), and it needs the body model files (see the smplfitter documentation).
"""

import argparse
import os
import sys

import numpy as np
import simplepyutils as spu

# PoseViz constructor arguments that a scene's display dict may suggest.
# Appearance only: camera control is deliberately NOT in this list — a scene
# file may store a viz-camera trajectory as a suggestion, but it must never
# be able to lock the interactive camera (that is the player's and the
# user's domain, see --free-camera and mouse takeover).
_DISPLAY_KEYS = (
    'show_ground_plane',
    'show_camera_wireframe',
    'show_field_of_view',
    'mesh_opacity',
    'show_image',
    'image_plane_distance',
    'high_quality',
    'camera_view_padding',
)


def main():
    parser = argparse.ArgumentParser(
        prog='poseviz', description='Play a .pvz scene file.'
    )
    parser.add_argument('scene', help='path to a .pvz scene file')
    parser.add_argument('--info', action='store_true', help='print a summary and exit')
    parser.add_argument('--paused', action='store_true', help='start paused')
    parser.add_argument('--loop', action='store_true', help='loop the scene')
    parser.add_argument('--out-video', metavar='PATH', help='record the playback to a video')
    parser.add_argument(
        '--speed', type=float, default=1.0, help='playback speed factor (default 1.0)'
    )
    parser.add_argument(
        '--skeleton', action='store_true', help='draw the body joints and bones too'
    )
    parser.add_argument(
        '--free-camera',
        action='store_true',
        help='ignore a stored visualization-camera trajectory',
    )
    args = parser.parse_args()

    from poseviz.scenefile import SceneReader

    reader = SceneReader(args.scene)
    if args.info:
        print_info(reader)
        return

    play(reader, args)


def play(reader, args):
    import framepump
    import poseviz

    body_model = load_body_model(reader.body_model)
    scale = reader.smpl_to_world_scale
    joint_edges = [
        [i, parent] for i, parent in enumerate(body_model.kintree_parents) if parent >= 0
    ]
    display = {k: v for k, v in reader.display.items() if k in _DISPLAY_KEYS}
    # camera mode is decided here, never by the scene file: follow a stored
    # viz-camera trajectory when one exists (mouse drag takes over), else free
    display['camera_type'] = (
        'original' if reader.has_viz_camera and not args.free_camera else 'free'
    )

    videos = [
        framepump.VideoFrames(reader.resolve_path(view['video']))
        if view['video'] is not None
        else None
        for view in reader.views
    ]

    with poseviz.PoseViz(
        joint_names=body_model.joint_names,
        joint_edges=joint_edges,
        n_views=len(reader.views),
        world_up=reader.world_up,
        ground_plane_height=reader.ground_plane_height,
        viz_fps=max(1, round(reader.fps * args.speed)),
        body_model_faces=body_model.faces,
        paused=args.paused,
        **display,
    ) as viz:
        if args.out_video:
            viz.new_sequence_output(args.out_video, fps=reader.fps)

        while True:
            play_once(reader, viz, videos, body_model, scale, args)
            if not args.loop:
                break
            viz.reinit_camera_view()

        if args.out_video:
            viz.finalize_sequence_output()
    if args.out_video:
        print(f'Wrote {args.out_video}')


def play_once(reader, viz, videos, body_model, scale, args):
    import poseviz

    video_iters = [iter(v) if v is not None else None for v in videos]
    last_cameras = [None] * len(reader.views)

    for i_frame in spu.progressbar(range(reader.n_frames), desc='Playing'):
        view_infos = []
        for i_view in range(len(reader.views)):
            frame = None
            if video_iters[i_view] is not None:
                frame = next(video_iters[i_view], None)
            camera = reader.view_camera(i_view, i_frame)
            if camera is None:
                camera = last_cameras[i_view]
            else:
                last_cameras[i_view] = camera
            if camera is None:
                raise ValueError(
                    f'View {i_view} has no camera (neither static nor trajectory) '
                    f'up to frame {i_frame}.'
                )
            if frame is not None:
                frame = np.ascontiguousarray(frame)
            view_infos.append(poseviz.ViewInfo(frame=frame, camera=camera))

        bodies = reader.bodies(i_frame)
        if bodies:
            pose = np.stack([b[1] for b in bodies])
            betas = np.stack(
                [
                    reader.betas.get(
                        b[0], np.zeros(reader.body_model['num_betas'], np.float32)
                    )
                    for b in bodies
                ]
            )
            trans = np.stack([b[2] for b in bodies])
            result = body_model(pose_rotvecs=pose, shape_betas=betas, trans=trans)
            view_infos[0].vertices = list(result['vertices'] * scale)
            if args.skeleton:
                view_infos[0].poses = list(result['joints'] * scale)

        viz.update_multiview(
            view_infos,
            viz_camera=None if args.free_camera else reader.viz_camera(i_frame),
        )


def load_body_model(spec):
    try:
        import smplfitter.np
    except ImportError:
        sys.exit(
            'The poseviz player needs smplfitter to evaluate SMPL parameters: '
            'pip install smplfitter (body model files are also required, '
            'see the smplfitter documentation).'
        )
    return smplfitter.np.get_cached_body_model(spec['model_name'], spec['gender'])


def print_info(reader):
    duration = reader.n_frames / reader.fps if reader.fps else 0
    print(f'{os.path.abspath(reader.path)}')
    print(
        f'{reader.n_frames} frames @ {reader.fps:g} fps ({duration:.1f} s), '
        f'world_up={tuple(reader.world_up)}'
    )
    bm = reader.body_model
    print(f'Body model: {bm["model_name"]} ({bm["gender"]}, {bm["num_betas"]} betas)')
    print(f'Tracks: {len(reader.betas)}, person-frames: {reader.n_person_frames}')
    print(f'Viz camera trajectory: {"yes" if reader.has_viz_camera else "no"}')
    print('Views:')
    for i, view in enumerate(reader.views):
        cam = 'static camera' if view['camera'] is not None else 'camera trajectory'
        video = view['video'] if view['video'] is not None else '(no video)'
        print(f'  [{i}] {view["name"]}: {video} — {cam}')


if __name__ == '__main__':
    main()
