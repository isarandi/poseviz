import time

import smpl.numpy

import poseviz

import numpy as np
import cameravision


def main():
    # Names of the body joints. Left joint names must start with 'l', right with 'r'.
    joint_names = ["l_wrist", "l_elbow"]

    # Joint index pairs specifying which ones should be connected with a line (i.e., the bones of
    # the body, e.g. wrist-elbow, elbow-shoulder)
    joint_edges = [[0, 1]]
    viz = poseviz.PoseViz(joint_names, joint_edges)

    # Iterate over the frames of e.g. a video
    for i in range(1):
        # Get the current frame
        frame = np.zeros([720, 1280, 3], np.uint8)

        # Make predictions here
        # ...

        # Update the visualization
        viz.update(
            frame=frame,
            boxes=np.array([[10, 20, 100, 100]], np.float32),
            poses=np.array([[[100, 100, 2000], [-100, 100, 2000]]], np.float32),
            camera=cameravision.Camera.from_fov(55, frame.shape[:2]),
        )


def main_smpl():
    smpl_canonical = np.load(
        "/work/sarandi/projects/localizerfields/canonical_vertices_smpl.npy"
    ) * [1,-1,-1] * 1000 + [0, 0, 3000]

    # Names of the body joints. Left joint names must start with 'l', right with 'r'.
    joint_names = ["l_wrist", "l_elbow"]

    # Joint index pairs specifying which ones should be connected with a line (i.e., the bones of
    # the body, e.g. wrist-elbow, elbow-shoulder)
    joint_edges = [[0, 1]]
    faces = smpl.numpy.get_cached_body_model("smpl", "neutral").faces
    with poseviz.PoseViz(joint_names, joint_edges, body_model_faces=faces) as viz:
        # Iterate over the frames of e.g. a video
        for i in range(1):
            # Get the current frame
            frame = np.zeros([720, 1280, 3], np.uint8)

            # Make predictions here
            # ...

            # Update the visualization
            viz.update(
                frame=frame,
                boxes=np.array([[10, 20, 100, 100]], np.float32),
                vertices=np.array([smpl_canonical + [500, 0, 5000]]),
                camera=cameravision.Camera.from_fov(55, frame.shape[:2], world_up=(0, 1, 0)),
            )

            viz.update(
                frame=frame,
                boxes=np.array([[10, 20, 100, 100]], np.float32),
                vertices=np.array(
                    [smpl_canonical + [-500, 0, 3000], smpl_canonical + [-500, 0, 5000]]
                ),
                camera=cameravision.Camera.from_fov(55, frame.shape[:2], world_up=(0, 1, 0)),
            )

        time.sleep(1000)


if __name__ == "__main__":
    main_smpl()
