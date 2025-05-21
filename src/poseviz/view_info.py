import dataclasses

import cameravision
import numpy as np
import poseviz.draw2d


@dataclasses.dataclass
class ViewInfo:
    """A container for storing information about a single view of a scene"""

    frame: any = None
    boxes: any = ()
    poses: any = ()
    camera: cameravision.Camera = None
    poses_true: any = ()
    poses_alt: any = ()
    vertices: any = ()
    vertices_true: any = ()
    vertices_alt: any = ()


def downscale_and_undistort_view_info(view_info, dst, index):
    scale_factor = np.array(dst.shape[:2], np.float32) / np.array(
        view_info.frame.shape[:2], np.float32
    )

    if view_info.frame.dtype == np.uint16:
        view_info.frame = np.ascontiguousarray(view_info.frame.view(np.uint8)[..., ::2])

    if np.any(scale_factor != 1):
        resize_dst = np.empty_like(dst) if view_info.camera.has_distortion() else dst

        view_info.frame = poseviz.draw2d.resize(view_info.frame, dst=resize_dst)
        if view_info.boxes is not None:
            view_info.boxes = np.asarray(view_info.boxes)
            view_info.boxes[:, :2] *= scale_factor
            view_info.boxes[:, 2:4] *= scale_factor
        view_info.camera = view_info.camera.scale_output(scale_factor, inplace=False)

    if view_info.camera.has_distortion():
        old_camera = view_info.camera
        view_info.camera = old_camera.undistort(
            alpha_balance=0.8, imshape=dst.shape, inplace=False
        )

        cameravision.reproject_image(
            view_info.frame,
            old_camera,
            view_info.camera,
            dst.shape,
            precomp_undist_maps=False,
            cache_maps=True,
            dst=dst,
        )

        if view_info.boxes is not None:
            for i in range(len(view_info.boxes)):
                view_info.boxes[i, :4] = cameravision.reproject_box(
                    view_info.boxes[i, :4], old_camera, view_info.camera
                )
    elif view_info.frame is not dst:
        np.copyto(dst, view_info.frame)

    view_info.frame = (dst.shape, dst.dtype, index)
    return view_info
