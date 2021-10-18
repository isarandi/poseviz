import numpy as np
from mayavi import mlab


class ConnectedPoints:
    def __init__(
            self, color, line_color, mode, scale_factor, opacity_points, opacity_lines,
            point_names, high_quality=False):
        self.coords = []
        self.n_rendered_points = 0
        self.color = color
        self.mode = mode
        self.scale_factor = scale_factor
        self.line_color = line_color
        self.opacity_points = opacity_points
        self.opacity_lines = opacity_lines
        self.is_initialized = False
        self.points = None
        self.tube = None
        self.edges = []
        self.point_name_components = []
        self.point_names = point_names
        self.high_quality = high_quality

    def clear(self):
        self.coords.clear()
        self.edges.clear()

    def add_points(self, coords, edges, ignore_isolated=True):
        if ignore_isolated:
            coords, edges = self.get_nonisolated(coords, edges)
        n_points = len(self.coords)
        self.coords.extend(coords)
        self.edges.extend([[n_points + i, n_points + j] for i, j in edges])

    def get_nonisolated(self, coords, edges):
        selected_coords = []
        new_point_ids = {}
        selected_edges = []

        for (i, j) in edges:
            if i not in new_point_ids:
                new_point_ids[i] = len(selected_coords)
                selected_coords.append(coords[i])
            if j not in new_point_ids:
                new_point_ids[j] = len(selected_coords)
                selected_coords.append(coords[j])
            selected_edges.append([new_point_ids[i], new_point_ids[j]])
        return selected_coords, selected_edges

    def update(self):
        if not self.is_initialized:
            return self.initial_update()
        c = np.asarray(self.coords)

        if not self.coords:
            c = c.reshape((0, 3))

        if len(self.coords) == self.n_rendered_points:
            self.points.mlab_source.set(x=c[:, 0], y=c[:, 1], z=c[:, 2])
        else:
            self.n_rendered_points = len(self.coords)
            self.points.mlab_source.reset(x=c[:, 0], y=c[:, 1], z=c[:, 2])
            self.points.mlab_source.dataset.lines = self.edges

    def initial_update(self):
        if not self.coords:
            return
        c = np.asarray(self.coords)

        self.points = mlab.points3d(
            *c.T, scale_factor=self.scale_factor, color=self.color,
            opacity=self.opacity_points,
            mode=self.mode, resolution=8 if self.high_quality else 4, scale_mode='vector',
            reset_zoom=False)
        self.points.mlab_source.dataset.lines = self.edges
        tube = mlab.pipeline.tube(
            self.points, tube_radius=self.scale_factor / 3,
            tube_sides=6 if self.high_quality else 3)
        mlab.pipeline.surface(
            tube, color=self.line_color, opacity=self.opacity_lines, reset_zoom=False)

        # for i, (point, name, color) in enumerate(
        #         zip(c, self.point_names, colors.cycle_over_colors())):
        #     self.point_name_components.append(mlab.text3d(
        #         *point, f'{i}', scale=0.02, color=color))

        self.is_initialized = True
