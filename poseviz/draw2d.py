import cv2
import numpy as np


def rounded_int_tuple(p):
    return tuple(np.round(p).astype(int))


def line(im, p1, p2, *args, **kwargs):
    cv2.line(im, rounded_int_tuple(p1), rounded_int_tuple(p2), *args, **kwargs)


def draw_stick_figure_2d_inplace(im, coords, joint_edges, thickness=3, color=None):
    for i_joint1, i_joint2 in joint_edges:
        relevant_coords = coords[[i_joint1, i_joint2]]
        if not np.isnan(relevant_coords).any() and not np.isclose(0, relevant_coords).any():
            line(
                im, coords[i_joint1], coords[i_joint2],
                color=color, thickness=thickness, lineType=cv2.LINE_AA)


def rectangle(im, pt1, pt2, color, thickness):
    cv2.rectangle(im, rounded_int_tuple(pt1), rounded_int_tuple(pt2), color, thickness)


def draw_box(im, box, color=(255, 0, 0), thickness=5):
    box = np.array(box)
    rectangle(im, box[:2], box[:2] + box[2:4], color, thickness)
