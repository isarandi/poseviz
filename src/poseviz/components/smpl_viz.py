import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab

import poseviz.mayavi_util


class SMPLViz:
    def __init__(self, color, faces, add_wireframe=False, colormap="viridis"):
        self.opacity = 0.6
        self.faces = faces
        self.is_initialized = False
        self.n_rendered_verts = 0
        self.color = color
        self.add_wireframe = add_wireframe
        self.colormap = colormap

    def update(self, poses):
        if not self.is_initialized:
            return self.initial_update(poses)

        mayavi_poses = [poseviz.mayavi_util.world_to_mayavi(pose[:, :3]) for pose in poses]
        uncerts = [pose[:, 3] if pose.shape[1] == 4 else None for pose in poses]

        if len(mayavi_poses) == 0:
            c = np.array([[6, -3, 2]], np.float32)
            u = np.array([0], np.float32)
        else:
            c = np.concatenate(mayavi_poses, axis=0)
            if uncerts[0] is not None:
                u = np.concatenate(uncerts, axis=0)
            else:
                u = None

        if len(c) == self.n_rendered_verts:
            # Number of vertices is the same as prev frame, so we can just update the positions
            if u is not None:
                self.mesh.mlab_source.set(x=c[:, 0], y=c[:, 1], z=c[:, 2], scalars=u)
            else:
                self.mesh.mlab_source.set(x=c[:, 0], y=c[:, 1], z=c[:, 2])
        else:
            self.n_rendered_verts = len(c)

            if len(mayavi_poses) == 0:
                triangles = np.zeros([1, 3], np.int32)
            else:
                num_vertices = poses[0].shape[0]
                triangles = np.concatenate(
                    [np.asarray(self.faces) + i * num_vertices for i in range(len(mayavi_poses))],
                    axis=0,
                )

            if u is not None:
                self.mesh.mlab_source.reset(
                    x=c[:, 0], y=c[:, 1], z=c[:, 2], triangles=triangles, scalars=u
                )
            else:
                self.mesh.mlab_source.reset(x=c[:, 0], y=c[:, 1], z=c[:, 2], triangles=triangles)

    def initial_update(self, poses):
        if len(poses) == 0:
            return

        mayavi_poses = [poseviz.mayavi_util.world_to_mayavi(pose[:, :3]) for pose in poses]
        uncerts = [pose[:, 3] if pose.shape[1] == 4 else None for pose in poses]

        num_vertices = poses[0].shape[0]
        triangles = np.concatenate(
            [np.asarray(self.faces) + i * num_vertices for i in range(len(mayavi_poses))], axis=0
        )
        c = np.concatenate(mayavi_poses, axis=0)
        if uncerts[0] is not None:
            u = np.concatenate(uncerts, axis=0)
        else:
            u = None

        if u is not None:
            self.mesh = mlab.triangular_mesh(
                *c.T,
                triangles,
                scalars=u,
                colormap="cool",
                vmin=0,
                vmax=0.1,
                opacity=self.opacity,
                representation="surface",
                reset_zoom=False,
            )

            cmap = plt.get_cmap(self.colormap)
            cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
            self.mesh.module_manager.scalar_lut_manager.lut.table = cmaplist
        else:
            self.mesh = mlab.triangular_mesh(
                *c.T,
                triangles,
                color=self.color,
                opacity=self.opacity,
                representation="surface",
                reset_zoom=False,
            )

        attrs = dict(backface_culling=True)

        if self.add_wireframe:
            attrs.update(edge_visibility=True, edge_color=(0.0, 0.0, 0.0), line_width=0.1)

        for key, value in attrs.items():
            setattr(self.mesh.actor.property, key, value)

        self.is_initialized = True
        self.n_rendered_meshes = len(mayavi_poses)

    def remove(self):
        if self.is_initialized:
            self.mesh.remove()
            self.is_initialized = False
