import dataclasses

import cameralib
import numpy as np

import poseviz.draw2d


@dataclasses.dataclass
class ViewInfo:
    frame: any = None
    boxes: any = ()
    poses: any = ()
    camera: cameralib.Camera = None
    poses_true: any = ()
    poses_alt: any = ()
    vertices: any = ()
    vertices_true: any = ()
    vertices_alt: any = ()

    def downscale(self, downscale_factor):
        if downscale_factor != 1:
            self.frame = poseviz.draw2d.resize_by_factor(self.frame, 1 / downscale_factor)
            self.boxes = np.asarray(self.boxes) / downscale_factor
            self.camera = self.camera.scale_output(1 / downscale_factor, inplace=False)

    def undistort(self):
        if not self.camera.has_distortion():
            return self

        old_camera = self.camera
        self.camera = old_camera.undistort(
            alpha_balance=0.8, imshape=self.frame.shape, inplace=False)
        self.frame = cameralib.reproject_image(
            self.frame, old_camera, self.camera, self.frame.shape)

        if self.boxes is not None:
            for i in range(len(self.boxes)):
                self.boxes[i, :4] = cameralib.reproject_box(
                    self.boxes[i, :4], old_camera, self.camera)

    def convert_fields_to_numpy(self):
        for field in dataclasses.fields(self):
            setattr(self, field.name, tf_to_numpy(getattr(self, field.name)))
        return self


def downscale_and_undistort_view_info(view_info, downscale_factor):
    view_info.downscale(downscale_factor)
    view_info.undistort()
    return view_info


def tf_to_numpy(x):
    try:
        import tensorflow as tf
    except ImportError:
        return x

    if isinstance(x, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        return x.numpy()
    return x
