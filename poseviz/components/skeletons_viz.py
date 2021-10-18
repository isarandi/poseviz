import poseviz.components.connected_points_viz
import poseviz.mayavi_util


class SkeletonsViz:
    def __init__(self, joint_info, left_color, mid_color, right_color, point_color, opacity,
                 high_quality=False):
        op = 0.7
        joint_names, joint_edges = joint_info
        edge_colors = dict(left=left_color, mid=mid_color, right=right_color)
        self.pointsets = {
            side: poseviz.components.connected_points_viz.ConnectedPoints(
                point_color, edge_colors[side], 'sphere', opacity, op, op, joint_names,
                high_quality)
            for side in ('left', 'mid', 'right')}

        def edge_side(edge):
            n1 = joint_names[edge[0]].lower()
            n2 = joint_names[edge[1]].lower()
            if n1.startswith('l') or n2.startswith('l'):
                return 'left'
            elif n1.startswith('r') or n2.startswith('r'):
                return 'right'
            else:
                return 'mid'

        self.edgesets = {
            side: [e for e in joint_edges if edge_side(e) == side]
            for side in ('left', 'mid', 'right')}

    def update(self, poses):
        mayavi_poses = [poseviz.mayavi_util.world_to_mayavi(pose) for pose in poses]
        for side, pointset in self.pointsets.items():
            pointset.clear()
            for coords in mayavi_poses:
                pointset.add_points(coords, self.edgesets[side])
            pointset.update()
