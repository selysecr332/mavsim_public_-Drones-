# rrt straight line path planner for mavsim_python
import numpy as np
from message_types.msg_waypoints import MsgWaypoints


class RRTStraightLine:
    def __init__(self):
        self.segment_length = 300

    def update(self, start_pose, end_pose, Va, world_map, radius):
        tree = MsgWaypoints()
        tree.type = 'fillet'

        tree.add(start_pose, Va, np.inf, 0, 0, 0)

        if (distance(start_pose, end_pose) < self.segment_length
                and not collision(start_pose, end_pose, world_map)):
            tree.add(end_pose, Va, np.inf, distance(start_pose, end_pose), 0, 1)
        else:
            num_paths = 0
            iteration = 0
            max_iterations = 1000
            while num_paths < 3 and iteration < max_iterations:
                flag = self.extend_tree(tree, end_pose, Va, world_map)
                num_paths += flag
                iteration += 1

        waypoints_not_smoothed = find_minimum_path(tree, end_pose)
        if waypoints_not_smoothed.num_waypoints < 3:
            waypoints_not_smoothed = _fallback_path(start_pose, end_pose, Va)
            tree = waypoints_not_smoothed

        waypoints = smooth_path(waypoints_not_smoothed, world_map)
        if waypoints.num_waypoints < 3:
            waypoints = waypoints_not_smoothed

        self.waypoints_not_smoothed = waypoints_not_smoothed
        self.tree = tree
        return waypoints

    def extend_tree(self, tree, end_pose, Va, world_map):
        pd = end_pose.item(2)
        random_pose = random_pose_fn(world_map, pd)

        nearest_idx = 0
        min_dist = np.inf
        for i in range(tree.num_waypoints):
            node = column(tree.ned, i)
            d = np.linalg.norm(node - random_pose)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        node_near = column(tree.ned, nearest_idx)
        direction = random_pose - node_near
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return 0
        direction = direction / dist

        if dist > self.segment_length:
            new_pose = node_near + self.segment_length * direction
        else:
            new_pose = random_pose

        if collision(node_near, new_pose, world_map):
            return 0

        cost = tree.cost[nearest_idx] + np.linalg.norm(new_pose - node_near)
        tree.add(new_pose, Va, np.inf, cost, nearest_idx, 0)
        new_idx = tree.num_waypoints - 1

        if (distance(new_pose, end_pose) < self.segment_length
                and not collision(new_pose, end_pose, world_map)):
            cost_to_goal = cost + distance(new_pose, end_pose)
            tree.add(end_pose, Va, np.inf, cost_to_goal, new_idx, 1)
            return 1
        return 0


def _fallback_path(start_pose, end_pose, Va):
    waypoints = MsgWaypoints()
    waypoints.type = 'fillet'
    waypoints.add(start_pose, Va, np.inf, 0, 0, 0)
    mid = (start_pose + end_pose) / 2.
    waypoints.add(mid, Va, np.inf, 0, 0, 0)
    waypoints.add(end_pose, Va, np.inf, 0, 1, 1)
    return waypoints


def smooth_path(waypoints, world_map):
    if waypoints.num_waypoints == 0:
        return waypoints

    smooth_indices = [0]
    i = 0
    while i < waypoints.num_waypoints - 1:
        j = waypoints.num_waypoints - 1
        while j > i + 1:
            if not collision(column(waypoints.ned, i), column(waypoints.ned, j), world_map):
                break
            j -= 1
        smooth_indices.append(j)
        i = j

    smooth_waypoints = MsgWaypoints()
    smooth_waypoints.type = waypoints.type
    for idx in smooth_indices:
        smooth_waypoints.add(
            column(waypoints.ned, idx),
            waypoints.airspeed[idx],
            waypoints.course[idx],
            0, 0, 0,
        )
    return smooth_waypoints


def find_minimum_path(tree, end_pose):
    connecting_nodes = []
    for i in range(tree.num_waypoints):
        if tree.connect_to_goal.item(i) == 1:
            connecting_nodes.append(i)
    if len(connecting_nodes) == 0:
        return MsgWaypoints()

    idx = int(np.argmin(tree.cost[connecting_nodes]))
    path = [connecting_nodes[idx]]
    parent_node = int(tree.parent.item(connecting_nodes[idx]))
    while parent_node >= 1:
        path.insert(0, parent_node)
        parent_node = int(tree.parent.item(parent_node))
    path.insert(0, 0)

    waypoints = MsgWaypoints()
    waypoints.type = tree.type
    for i in path:
        waypoints.add(column(tree.ned, i),
                      tree.airspeed.item(i),
                      np.inf, np.inf, np.inf, np.inf)
    waypoints.add(end_pose,
                  tree.airspeed[-1],
                  np.inf, np.inf, np.inf, np.inf)
    return waypoints


def random_pose_fn(world_map, pd):
    pn = world_map.city_width * np.random.rand()
    pe = world_map.city_width * np.random.rand()
    return np.array([[pn], [pe], [pd]])


def distance(start_pose, end_pose):
    return np.linalg.norm(start_pose - end_pose)


def collision(start_pose, end_pose, world_map):
    points = points_along_path(start_pose, end_pose, 100)
    for i in range(points.shape[1]):
        if height_above_ground(world_map, column(points, i)) <= 0:
            return True
    return False


def height_above_ground(world_map, point):
    point_height = -point.item(2)
    tmp = np.abs(point.item(0) - world_map.building_north)
    d_n = np.min(tmp)
    idx_n = int(np.argmin(tmp))
    tmp = np.abs(point.item(1) - world_map.building_east)
    d_e = np.min(tmp)
    idx_e = int(np.argmin(tmp))
    if (d_n < world_map.building_width) and (d_e < world_map.building_width):
        map_height = world_map.building_height[idx_n, idx_e]
    else:
        map_height = 0
    return point_height - map_height


def points_along_path(start_pose, end_pose, N):
    points = start_pose
    q = (end_pose - start_pose)
    L = np.linalg.norm(q)
    if L < 1e-6:
        return points
    q = q / L
    w = start_pose.copy()
    for i in range(1, N):
        w = w + (L / N) * q
        points = np.append(points, w, axis=1)
    return points


def column(A, i):
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col
