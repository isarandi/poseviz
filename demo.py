import poseviz
import numpy as np


def main():
    # Names of the body joints. Left joint names must start with 'l', right with 'r'.
    joint_names = ['l_wrist', 'l_elbow']

    # Joint index pairs specifying which ones should be connected with a line (i.e., the bones of
    # the body, e.g. wrist-elbow, elbow-shoulder)
    joint_edges = [[0, 1]]
    viz = poseviz.PoseViz(joint_names, joint_edges)

    # Iterate over the frames of e.g. a video
    for i in range(1):
        # Get the current frame
        frame = np.zeros([512, 512, 3], np.uint8)

        # Make predictions here
        # ...

        # Update the visualization
        viz.update(
            frame=frame,
            boxes=np.array([[10, 20, 100, 100]], np.float32),
            poses=np.array([[[100, 100, 2000], [-100, 100, 2000]]], np.float32),
            camera=poseviz.cameralib.Camera.from_fov(55, frame.shape[:2]))


if __name__ == '__main__':
    main()
