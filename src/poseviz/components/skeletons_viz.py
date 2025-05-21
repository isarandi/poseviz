import poseviz.components.connected_points_viz
import poseviz.mayavi_util


class SkeletonsViz:
    def __init__(
        self,
        joint_info,
        left_color,
        mid_color,
        right_color,
        point_color,
        scale_factor,
        high_quality=False,
        opacity=0.7,
    ):

        joint_names, joint_edges = joint_info
        edge_colors = dict(left=left_color, mid=mid_color, right=right_color)
        sides = ("left", "mid", "right")
        self.pointsets = {
            side: poseviz.components.connected_points_viz.ConnectedPoints(
                point_color,
                edge_colors[side],
                "sphere",
                scale_factor,
                opacity,
                opacity,
                joint_names,
                high_quality,
            )
            for side in sides
        }

        def edge_side(edge):
            s1 = joint_side(edge[0])
            s2 = joint_side(edge[1])
            if s1 == "left" or s2 == "left":
                return "left"
            elif s1 == "right" or s2 == "right":
                return "right"
            else:
                return "mid"

        def joint_side(joint):
            name = joint_names[joint].lower()
            if name.startswith("l"):
                return "left"
            elif name.startswith("r"):
                return "right"
            else:
                return "mid"

        self.edges_per_side = {
            side: [e for e in joint_edges if edge_side(e) == side] for side in sides
        }

        self.indices_per_side = {
            side: sorted(
                set([index for edge in self.edges_per_side[side] for index in edge])
                | set(
                    [index for index, name in enumerate(joint_names) if joint_side(index) == side]
                )
            )
            for side in sides
        }

        def index_within_side(global_index, side):
            return self.indices_per_side[side].index(global_index)

        self.edges_within_side = {
            side: [
                (index_within_side(i, side), index_within_side(j, side))
                for i, j in self.edges_per_side[side]
            ]
            for side in sides
        }

    def update(self, poses):
        mayavi_poses = [poseviz.mayavi_util.world_to_mayavi(pose) for pose in poses]
        for side, pointset in self.pointsets.items():
            pointset.clear()
            for coords in mayavi_poses:
                pointset.add_points(
                    coords[self.indices_per_side[side]],
                    self.edges_within_side[side],
                    show_isolated_points=True,
                )
            pointset.update()

    def remove(self):
        for pointset in self.pointsets.values():
            pointset.remove()
