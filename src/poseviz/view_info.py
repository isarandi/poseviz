import dataclasses

import deltacamera
import numpy as np
import poseviz.draw2d


@dataclasses.dataclass
class ViewInfo:
    """A container for storing information about a single view of a scene"""

    frame: object = None
    boxes: tuple = ()
    poses: tuple = ()
    camera: deltacamera.Camera = None
    poses_true: tuple = ()
    poses_alt: tuple = ()
    vertices: tuple = ()
    vertices_true: tuple = ()
    vertices_alt: tuple = ()


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
        view_info.camera = view_info.camera.image_scaled(scale_factor)

    if view_info.camera.has_distortion():
        old_camera = view_info.camera
        view_info.camera, _, _ = old_camera.undistorted_with_optimal_intrinsics(
            alpha_balance=0.8, imshape_distorted=dst.shape
        )

        deltacamera.reproject_image(
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
                view_info.boxes[i, :4] = deltacamera.reproject_box(
                    view_info.boxes[i, :4], old_camera, view_info.camera
                )
    elif view_info.frame is not dst:
        np.copyto(dst, view_info.frame)

    view_info.frame = (dst.shape, dst.dtype, index)
    return view_info


def gpu_downscale_and_undistort(view_info, downscale_factor):
    """GPU version of downscale_and_undistort_view_info.

    Takes a ViewInfo with a CUDA tensor (or DLPack) frame, performs resize+undistort
    on GPU via deltacamera.pt.reproject_image (single grid_sample kernel), and returns
    a ViewInfo where frame is a CUDA tensor (H, W, 3) uint8.
    """
    import torch
    import deltacamera.pt as dpt

    frame = view_info.frame
    if not isinstance(frame, torch.Tensor):
        frame = torch.from_dlpack(frame)
    if not frame.is_contiguous():
        frame = frame.contiguous()

    h, w = frame.shape[:2]
    new_h = round(h / downscale_factor)
    new_w = round(w / downscale_factor)

    gpu_cam = dpt.Camera.from_cpu(view_info.camera, device=frame.device)
    new_cam = gpu_cam.image_resized((new_h, new_w))
    if gpu_cam.has_distortion():
        new_cam = new_cam.undistorted()

    # reproject_image expects (B, C, H, W) float
    frame_bchw = frame.permute(2, 0, 1).unsqueeze(0).float()
    result = dpt.reproject_image(frame_bchw, gpu_cam, new_cam)
    result = result.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().contiguous()

    # Scale boxes
    if view_info.boxes is not None and len(view_info.boxes) > 0:
        sy, sx = new_h / h, new_w / w
        boxes = np.asarray(view_info.boxes, dtype=np.float32)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        view_info.boxes = boxes

    # Convert camera back to CPU deltacamera.Camera
    view_info.camera = deltacamera.Camera(
        rot_world_to_cam=new_cam.R.cpu().numpy(),
        optical_center=new_cam.t.cpu().numpy(),
        intrinsic_matrix=new_cam.K.cpu().numpy(),
        world_up=view_info.camera.world_up,
        image_shape=(new_h, new_w),
    )
    view_info.frame = result  # CUDA tensor (H, W, 3) uint8
    return view_info
