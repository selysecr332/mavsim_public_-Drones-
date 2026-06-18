# rrt dubins path planner for mavsim_python
import numpy as np
from message_types.msg_waypoints import MsgWaypoints
from planners.dubins_parameters import DubinsParameters


class RRTDubins:
    def __init__(self):
        self.segment_length = 450
        self.dubins_path = DubinsParameters()

    def update(self, start_pose, end_pose, Va, world_map, radius):
        self.segment_length = 4 * radius
        tree = MsgWaypoints()
        tree.type = 'dubins'

        tree.add(start_pose[0:3], Va, start_pose.item(3), 0, 0, 0)

        if (distance(start_pose, end_pose) < self.segment_length
                and not self.collision(start_pose, end_pose, world_map, radius)
                and np.linalg.norm(start_pose[0:3] - end_pose[0:3]) >= 2 * radius):
            tree.add(end_pose[0:3], Va, end_pose.item(3),
                     distance(start_pose, end_pose), 0, 1)
        else:
            num_paths = 0
            iteration = 0
            max_iterations = 1000
            while num_paths < 3 and iteration < max_iterations:
                flag = self.extendTree(tree, end_pose, Va, world_map, radius)
                num_paths += flag
                iteration += 1

        waypoints_not_smooth = findMinimumPath(tree, end_pose)
        if waypoints_not_smooth.num_waypoints < 3:
            waypoints_not_smooth = _fallback_path(start_pose, end_pose, Va, radius)
            tree = waypoints_not_smooth

        waypoints = self.smoothPath(waypoints_not_smooth, world_map, radius)
        if waypoints.num_waypoints < 3:
            waypoints = waypoints_not_smooth

        self.waypoint_not_smooth = waypoints_not_smooth
        self.tree = tree
        return waypoints

    def extendTree(self, tree, end_pose, Va, world_map, radius):
        pd = end_pose.item(2)
        random_pose = randomPose(world_map, pd)
        random_pose[3, 0] = 2 * np.pi * np.random.rand()

        nearest_idx = 0
        min_dist = np.inf
        for i in range(tree.num_waypoints):
            node = column(tree.ned, i)
            d = np.linalg.norm(node[0:2] - random_pose[0:2])
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        pose_near = np.vstack((
            column(tree.ned, nearest_idx),
            [[tree.course[nearest_idx]]],
        ))

        if np.linalg.norm(pose_near[0:3] - random_pose[0:3]) < 2 * radius:
            return 0

        self.dubins_path.update(
            pose_near[0:3], pose_near.item(3),
            random_pose[0:3], random_pose.item(3), radius)

        if self.dubins_path.length > self.segment_length:
            new_pose = self._pose_at_length(self.segment_length)
        else:
            new_pose = random_pose.copy()

        if self.collision(pose_near, new_pose, world_map, radius):
            return 0

        cost = tree.cost[nearest_idx] + min(self.segment_length, self.dubins_path.length)
        tree.add(new_pose[0:3], Va, new_pose.item(3), cost, nearest_idx, 0)
        new_idx = tree.num_waypoints - 1

        if (distance(new_pose, end_pose) < self.segment_length
                and not self.collision(new_pose, end_pose, world_map, radius)
                and np.linalg.norm(new_pose[0:3] - end_pose[0:3]) >= 2 * radius):
            cost_to_goal = cost + distance(new_pose, end_pose)
            tree.add(end_pose[0:3], Va, end_pose.item(3), cost_to_goal, new_idx, 1)
            return 1
        return 0

    def collision(self, start_pose, end_pose, world_map, radius):
        if np.linalg.norm(start_pose[0:3] - end_pose[0:3]) < 2 * radius:
            return True
        self.dubins_path.update(
            start_pose[0:3], start_pose.item(3),
            end_pose[0:3], end_pose.item(3), radius)
        if np.isinf(self.dubins_path.length):
            return True
        points = self.dubins_path.compute_points()
        for k in range(points.shape[0]):
            pt = np.array([[points[k, 0]], [points[k, 1]], [points[k, 2]]])
            if heightAboveGround(world_map, pt) <= 0:
                return True
        return False

    def smoothPath(self, waypoints, world_map, radius):
        if waypoints.num_waypoints == 0:
            return waypoints

        smooth_indices = [0]
        i = 0
        while i < waypoints.num_waypoints - 1:
            j = waypoints.num_waypoints - 1
            while j > i + 1:
                pose_i = np.vstack((
                    column(waypoints.ned, i),
                    [[waypoints.course[i]]],
                ))
                pose_j = np.vstack((
                    column(waypoints.ned, j),
                    [[waypoints.course[j]]],
                ))
                if not self.collision(pose_i, pose_j, world_map, radius):
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

    def _pose_at_length(self, length):
        points = self.dubins_path.compute_points()
        if points.shape[0] < 2:
            return np.array([[self.dubins_path.p_s.item(0)],
                             [self.dubins_path.p_s.item(1)],
                             [self.dubins_path.p_s.item(2)],
                             [self.dubins_path.chi_s]])

        total = 0.
        for k in range(1, points.shape[0]):
            seg = np.linalg.norm(points[k] - points[k - 1])
            if total + seg >= length:
                t = (length - total) / seg if seg > 0 else 0.
                pos = (1 - t) * points[k - 1] + t * points[k]
                chi = np.arctan2(points[k, 1] - points[k - 1, 1],
                                 points[k, 0] - points[k - 1, 0])
                return np.array([[pos[0]], [pos[1]], [pos[2]], [chi]])
            total += seg

        return np.array([[points[-1, 0]], [points[-1, 1]], [points[-1, 2]],
                         [self.dubins_path.chi_e]])


def _fallback_path(start_pose, end_pose, Va, radius):
    waypoints = MsgWaypoints()
    waypoints.type = 'dubins'
    waypoints.add(start_pose[0:3], Va, start_pose.item(3), 0, 0, 0)
    mid = (start_pose[0:3] + end_pose[0:3]) / 2.
    chi_mid = np.arctan2(end_pose.item(1) - start_pose.item(1),
                         end_pose.item(0) - start_pose.item(0))
    waypoints.add(mid, Va, chi_mid, 0, 0, 0)
    waypoints.add(end_pose[0:3], Va, end_pose.item(3), 0, 1, 1)
    return waypoints


def findMinimumPath(tree, end_pose):
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
                      tree.course.item(i),
                      np.inf, np.inf, np.inf)
    waypoints.add(end_pose[0:3],
                  tree.airspeed[-1],
                  end_pose.item(3),
                  np.inf, np.inf, np.inf)
    return waypoints


def distance(start_pose, end_pose):
    return np.linalg.norm(start_pose[0:3] - end_pose[0:3])


def heightAboveGround(world_map, point):
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


def randomPose(world_map, pd):
    pn = world_map.city_width * np.random.rand()
    pe = world_map.city_width * np.random.rand()
    chi = 0.
    pose = np.array([[pn], [pe], [pd], [chi]])
    return pose


def column(A, i):
    tmp = A[:, i]
    col = tmp.reshape(A.shape[0], 1)
    return col
