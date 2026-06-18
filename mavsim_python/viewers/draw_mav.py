"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        4/15/2019 - BGM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh


class DrawMav:
    def __init__(self, state, window, scale=1):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_index, self.mav_meshColors = self.get_mav_points()
        self.mav_body = self.add_object(
            self.mav_points,
            self.mav_index,
            self.mav_meshColors,
            R_bi,
            mav_position)
        window.addItem(self.mav_body)  # add mav to plot

    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        self.mav_body = self.update_object(
            self.mav_body,
            self.mav_points,
            self.mav_index,
            self.mav_meshColors,
            R_bi,
            mav_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,
            vertexColors=colors,
            drawEdges=True,
            smooth=False,
            computeNormals=False)
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def get_mav_points(self):
        """
        Points that define the mav, and the colors of the triangular mesh.
        Define the points on the aircraft following Figure 2.13.
        """
        fuse_l1 = 7
        fuse_l2 = 4
        fuse_l3 = 15
        fuse_w = 2
        wing_l = 6
        wing_w = 20
        tail_h = 3
        tailwing_w = 10
        tailwing_l = 3

        points = self.unit_length * np.array([
            [fuse_l1, 0, 0],
            [fuse_l2, -fuse_w / 2, -fuse_w / 2],
            [fuse_l2, fuse_w / 2, -fuse_w / 2],
            [fuse_l2, fuse_w / 2, fuse_w / 2],
            [fuse_l2, -fuse_w / 2, fuse_w / 2],
            [-fuse_l3, 0, 0],
            [0, wing_w / 2, 0],
            [-wing_l, wing_w / 2, 0],
            [-wing_l, -wing_w / 2, 0],
            [0, -wing_w / 2, 0],
            [-fuse_l3 + tailwing_l, tailwing_w / 2, 0],
            [-fuse_l3, tailwing_w / 2, 0],
            [-fuse_l3, -tailwing_w / 2, 0],
            [-fuse_l3 + tailwing_l, -tailwing_w / 2, 0],
            [-fuse_l3 + tailwing_l, 0, 0],
            [-fuse_l3 + tailwing_l, 0, -tail_h],
            [-fuse_l3, 0, -tail_h],
        ]).T

        index = np.array([
            [0, 1, 2],   # nose-top
            [0, 2, 3],   # nose-left
            [0, 3, 4],   # nose-bottom
            [0, 4, 1],   # nose-right
            [1, 2, 5],   # fuselage-top
            [2, 5, 3],   # fuselage-left
            [3, 5, 4],   # fuselage-bottom
            [1, 4, 5],   # fuselage-right
            [6, 7, 8],   # wing 1
            [6, 8, 9],   # wing 2
            [10, 11, 12],  # tailwing 1
            [10, 12, 13],  # tailwing 2
            [5, 14, 16],   # tail
        ])

        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow
        meshColors[1] = yellow
        meshColors[2] = yellow
        meshColors[3] = yellow
        meshColors[4] = blue
        meshColors[5] = blue
        meshColors[6] = red
        meshColors[7] = blue
        meshColors[8] = green
        meshColors[9] = green
        meshColors[10] = green
        meshColors[11] = green
        meshColors[12] = blue
        return points, index, meshColors
