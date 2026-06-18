# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab
#     - Beard & McLain, PUP, 2012
import numpy as np


class DubinsParameters:
    '''
    Class that contains parameters for a Dubin's car path
    '''

    def __init__(self):
        self.p_s = np.inf * np.ones((3, 1))
        self.chi_s = np.inf
        self.p_e = np.inf * np.ones((3, 1))
        self.chi_e = np.inf
        self.radius = np.inf
        self.length = np.inf
        self.center_s = np.inf * np.ones((3, 1))
        self.dir_s = np.inf
        self.center_e = np.inf * np.ones((3, 1))
        self.dir_e = np.inf
        self.r1 = np.inf * np.ones((3, 1))
        self.n1 = np.inf * np.ones((3, 1))
        self.r2 = np.inf * np.ones((3, 1))
        self.r3 = np.inf * np.ones((3, 1))
        self.n3 = np.inf * np.ones((3, 1))

    def update(self,
               ps: np.ndarray,
               chis: float,
               pe: np.ndarray,
               chie: float,
               R: float):
        self.p_s = ps
        self.chi_s = chis
        self.p_e = pe
        self.chi_e = chie
        self.radius = R
        self.compute_parameters()

    def compute_parameters(self):
        ps = self.p_s
        pe = self.p_e
        chis = self.chi_s
        chie = self.chi_e
        R = self.radius
        ell = np.linalg.norm(ps[0:2] - pe[0:2])

        if ell < 2 * R:
            self.length = np.inf
            return

        e1 = np.array([[1.], [0.], [0.]])
        heading_s = np.array([[np.cos(chis)], [np.sin(chis)], [0.]])
        heading_e = np.array([[np.cos(chie)], [np.sin(chie)], [0.]])

        crs = ps + R * (rotz(np.pi / 2) @ heading_s)
        cls = ps + R * (rotz(-np.pi / 2) @ heading_s)
        cre = pe + R * (rotz(np.pi / 2) @ heading_e)
        cle = pe + R * (rotz(-np.pi / 2) @ heading_e)

        L1 = _length_rsr(R, crs, cre, chis, chie)
        L2 = _length_rsl(R, crs, cle, chis, chie)
        L3 = _length_lsr(R, cls, cre, chis, chie)
        L4 = _length_lsl(R, cls, cle, chis, chie)

        lengths = np.array([L1, L2, L3, L4])
        min_idx = int(np.argmin(lengths))
        self.length = lengths[min_idx]

        if min_idx == 0:
            cs, lams, ce, lame, w1, q1, w2 = _params_rsr(
                R, crs, cre, chis, chie, ps, pe, e1)
        elif min_idx == 1:
            cs, lams, ce, lame, w1, q1, w2 = _params_rsl(
                R, crs, cle, chis, chie, ps, pe, e1)
        elif min_idx == 2:
            cs, lams, ce, lame, w1, q1, w2 = _params_lsr(
                R, cls, cre, chis, chie, ps, pe, e1)
        else:
            cs, lams, ce, lame, w1, q1, w2 = _params_lsl(
                R, cls, cle, chis, chie, ps, pe, e1)

        self.center_s = cs
        self.dir_s = lams
        self.center_e = ce
        self.dir_e = lame
        self.r1 = w1
        self.n1 = q1
        self.r2 = w2
        self.r3 = pe
        self.n3 = rotz(chie) @ e1

    def compute_points(self):
        Del = 0.1

        th1 = np.arctan2(self.p_s.item(1) - self.center_s.item(1),
                         self.p_s.item(0) - self.center_s.item(0))
        th1 = mod(th1)
        th2 = np.arctan2(self.r1.item(1) - self.center_s.item(1),
                         self.r1.item(0) - self.center_s.item(0))
        th2 = mod(th2)
        th = th1
        theta_list = [th]
        if self.dir_s > 0:
            if th1 >= th2:
                while th < th2 + 2 * np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2 * np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)

        points = np.array([[self.center_s.item(0) + self.radius * np.cos(theta_list[0]),
                            self.center_s.item(1) + self.radius * np.sin(theta_list[0]),
                            self.center_s.item(2)]])
        for angle in theta_list:
            new_point = np.array([[self.center_s.item(0) + self.radius * np.cos(angle),
                                   self.center_s.item(1) + self.radius * np.sin(angle),
                                   self.center_s.item(2)]])
            points = np.concatenate((points, new_point), axis=0)

        sig = 0.
        while sig <= 1.:
            new_point = np.array([[(1 - sig) * self.r1.item(0) + sig * self.r2.item(0),
                                   (1 - sig) * self.r1.item(1) + sig * self.r2.item(1),
                                   (1 - sig) * self.r1.item(2) + sig * self.r2.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            sig += Del

        th2 = np.arctan2(self.p_e.item(1) - self.center_e.item(1),
                         self.p_e.item(0) - self.center_e.item(0))
        th2 = mod(th2)
        th1 = np.arctan2(self.r2.item(1) - self.center_e.item(1),
                         self.r2.item(0) - self.center_e.item(0))
        th1 = mod(th1)
        th = th1
        theta_list = [th]
        if self.dir_e > 0:
            if th1 >= th2:
                while th < th2 + 2 * np.pi - Del:
                    th += Del
                    theta_list.append(th)
            else:
                while th < th2 - Del:
                    th += Del
                    theta_list.append(th)
        else:
            if th1 <= th2:
                while th > th2 - 2 * np.pi + Del:
                    th -= Del
                    theta_list.append(th)
            else:
                while th > th2 + Del:
                    th -= Del
                    theta_list.append(th)
        for angle in theta_list:
            new_point = np.array([[self.center_e.item(0) + self.radius * np.cos(angle),
                                   self.center_e.item(1) + self.radius * np.sin(angle),
                                   self.center_e.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        return points


def _length_rsr(R, crs, cre, chis, chie):
    theta = np.arctan2(cre.item(1) - crs.item(1), cre.item(0) - crs.item(0))
    return (np.linalg.norm(crs[0:2] - cre[0:2])
            + R * fmod(2 * np.pi + fmod(theta - np.pi / 2, 2 * np.pi)
                       - fmod(chis - np.pi / 2, 2 * np.pi), 2 * np.pi)
            + R * fmod(2 * np.pi + fmod(chie - np.pi / 2, 2 * np.pi)
                       - fmod(theta - np.pi / 2, 2 * np.pi), 2 * np.pi))


def _length_lsl(R, cls, cle, chis, chie):
    theta = np.arctan2(cle.item(1) - cls.item(1), cle.item(0) - cls.item(0))
    return (np.linalg.norm(cls[0:2] - cle[0:2])
            + R * fmod(2 * np.pi - fmod(theta + np.pi / 2, 2 * np.pi)
                       + fmod(chis + np.pi / 2, 2 * np.pi), 2 * np.pi)
            + R * fmod(2 * np.pi - fmod(chie + np.pi / 2, 2 * np.pi)
                       + fmod(theta + np.pi / 2, 2 * np.pi), 2 * np.pi))


def _length_lsr(R, cls, cre, chis, chie):
    ell = np.linalg.norm(cre[0:2] - cls[0:2])
    if ell < 2 * R:
        return 1e8
    theta = np.arctan2(cre.item(1) - cls.item(1), cre.item(0) - cls.item(0))
    theta2 = np.arccos(np.clip(2 * R / ell, -1., 1.))
    return (np.sqrt(ell ** 2 - 4 * R ** 2)
            + R * fmod(2 * np.pi - fmod(theta + theta2, 2 * np.pi)
                       + fmod(chis + np.pi / 2, 2 * np.pi), 2 * np.pi)
            + R * fmod(2 * np.pi - fmod(theta + theta2 - np.pi, 2 * np.pi)
                       + fmod(chie - np.pi / 2, 2 * np.pi), 2 * np.pi))


def _length_rsl(R, crs, cle, chis, chie):
    ell = np.linalg.norm(cle[0:2] - crs[0:2])
    if ell < 2 * R:
        return 1e8
    theta = np.arctan2(cle.item(1) - crs.item(1), cle.item(0) - crs.item(0))
    asin_value = np.clip(2 * R / ell, -1., 1.)
    theta2 = theta - np.pi / 2 + np.arcsin(asin_value)
    return (np.sqrt(max(ell ** 2 - 4 * R ** 2, 0.))
            + R * fmod(2 * np.pi + fmod(theta2, 2 * np.pi)
                       - fmod(chis - np.pi / 2, 2 * np.pi), 2 * np.pi)
            + R * fmod(2 * np.pi + fmod(theta2 + np.pi, 2 * np.pi)
                       - fmod(chie + np.pi / 2, 2 * np.pi), 2 * np.pi))


def _params_rsr(R, crs, cre, chis, chie, ps, pe, e1):
    theta = np.arctan2(cre.item(1) - crs.item(1), cre.item(0) - crs.item(0))
    w1 = crs + R * (rotz(theta - np.pi / 2) @ e1)
    w2 = cre + R * (rotz(theta - np.pi / 2) @ e1)
    q1 = (w2 - w1) / np.linalg.norm(w2 - w1)
    return crs, 1., cre, 1., w1, q1, w2


def _params_rsl(R, crs, cle, chis, chie, ps, pe, e1):
    ell = np.linalg.norm(cle[0:2] - crs[0:2])
    theta = np.arctan2(cle.item(1) - crs.item(1), cle.item(0) - crs.item(0))
    theta2 = theta - np.pi / 2 + np.arcsin(np.clip(2 * R / ell, -1., 1.))
    w1 = crs + R * (rotz(theta2) @ e1)
    w2 = cle + R * (rotz(theta2 + np.pi) @ e1)
    q1 = (w2 - w1) / np.linalg.norm(w2 - w1)
    return crs, 1., cle, -1., w1, q1, w2


def _params_lsr(R, cls, cre, chis, chie, ps, pe, e1):
    ell = np.linalg.norm(cre[0:2] - cls[0:2])
    theta = np.arctan2(cre.item(1) - cls.item(1), cre.item(0) - cls.item(0))
    theta2 = np.arccos(np.clip(2 * R / ell, -1., 1.))
    w1 = cls + R * (rotz(theta + theta2) @ e1)
    w2 = cre + R * (rotz(-np.pi + theta + theta2) @ e1)
    q1 = (w2 - w1) / np.linalg.norm(w2 - w1)
    return cls, -1., cre, 1., w1, q1, w2


def _params_lsl(R, cls, cle, chis, chie, ps, pe, e1):
    theta = np.arctan2(cle.item(1) - cls.item(1), cle.item(0) - cls.item(0))
    w1 = cls + R * (rotz(theta + np.pi / 2) @ e1)
    w2 = cle + R * (rotz(theta + np.pi / 2) @ e1)
    q1 = (w2 - w1) / np.linalg.norm(w2 - w1)
    return cls, -1., cle, -1., w1, q1, w2


def rotz(theta: float):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


def mod(x: float):
    while x < 0:
        x += 2 * np.pi
    while x > 2 * np.pi:
        x -= 2 * np.pi
    return x


def fmod(x: float, y: float):
    return np.mod(x, y)
