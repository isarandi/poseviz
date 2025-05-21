import numpy as np
import poseviz.mayavi_util
import tvtk.api
from mayavi import mlab


def draw_checkerboard(ground_plane_height):
    i, j = np.mgrid[:40, :40]
    image = ((i + j) % 2 == 0).astype(np.uint8)
    image[image == 0] = 0
    image[image == 1] = 255 - 96

    # image[image == 0] = 96
    # image[image == 1] = 256-96
    image = np.expand_dims(image, -1) * np.ones(3)

    size = 2 * 19
    extent = [0, size, 0, size, 0, 0]
    viz_im = mlab.imshow(
        np.ones(image.shape[:2]), opacity=0.3, extent=extent, reset_zoom=False, interpolate=False
    )

    reshaped = image.reshape([-1, 3], order="F")
    colors = tvtk.api.tvtk.UnsignedCharArray()
    colors.number_of_components = 3
    colors.from_array(reshaped)
    viz_im.actor.input.point_data.scalars = colors
    viz_im.actor.orientation = [0, 0, 90]
    viz_im.actor.position = np.array([-1, 1, ground_plane_height * poseviz.mayavi_util.MM_TO_UNIT])
