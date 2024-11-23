import math
import numpy as np
from scipy.interpolate import interp1d
import pytransform3d.rotations as pr
import pytransform3d.batch_rotations as pbr
import pytransform3d.transformations as pt


EPSILON = 1e-10

try:
    from ..dmp_fast import obstacle_avoidance_acceleration_2d as obstacle_avoidance_acceleration_2d_fast
    obstacle_avoidance_acceleration_2d_fast_available = True
except ImportError:
    obstacle_avoidance_acceleration_2d_fast_available = False


def obstacle_avoidance_acceleration_2d(
        y, yd, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
    """Compute acceleration for obstacle avoidance in 2D.

    Parameters
    ----------
    y : array, shape (..., 2)
        Current position(s).

    yd : array, shape (..., 2)
        Current velocity / velocities.

    obstacle_position : array, shape (2,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Obstacle avoidance parameter.

    beta : float, optional (default: 20 / pi)
        Obstacle avoidance parameter.

    Returns
    -------
    cdd : array, shape (..., 2)
        Accelerations.
    """
    obstacle_diff = obstacle_position - y
    pad_width = ([[0, 0]] * (y.ndim - 1)) + [[0, 1]]
    obstacle_diff_0 = np.pad(obstacle_diff, pad_width, mode="constant",
                             constant_values=0.0)
    yd_0 = np.pad(yd, pad_width, mode="constant", constant_values=0.0)
    r = 0.5 * np.pi * pbr.norm_vectors(np.cross(obstacle_diff_0, yd_0))
    R = pbr.matrices_from_compact_axis_angles(r)[..., :2, :2]
    theta_nom = np.einsum(
        "ni,ni->n", obstacle_diff.reshape(-1, 2), yd.reshape(-1, 2))
    shape = y.shape[:-1]
    if shape:
        theta_nom = theta_nom.reshape(*shape)
    theta_denom = (np.linalg.norm(obstacle_diff, axis=-1)
                   * np.linalg.norm(yd, axis=-1) + EPSILON)
    theta = np.arccos(theta_nom / theta_denom)
    rotated_velocity = np.einsum(
        "nij,nj->ni", R.reshape(-1, 2, 2), yd.reshape(-1, 2))
    if shape:
        rotated_velocity = rotated_velocity.reshape(*(shape + (2,)))
    cdd = (gamma * rotated_velocity
           * (theta * np.exp(-beta * theta))[..., np.newaxis])
    return np.squeeze(cdd)


class CouplingTermObstacleAvoidance2D:
    """Coupling term for obstacle avoidance in 2D.

    For :class:`DMP` and :class:`DMPWithFinalVelocity`.

    This is the simplified 2D implementation of
    :class:`CouplingTermObstacleAvoidance3D`.

    Parameters
    ----------
    obstacle_position : array, shape (2,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Parameter of obstacle avoidance.

    beta : float, optional (default: 20 / pi)
        Parameter of obstacle avoidance.
    """
    def __init__(self, obstacle_position, gamma=1000.0, beta=20.0 / math.pi,
                 fast=False):
        self.obstacle_position = obstacle_position
        self.gamma = gamma
        self.beta = beta
        if fast and obstacle_avoidance_acceleration_2d_fast_available:
            self.step_function = obstacle_avoidance_acceleration_2d_fast
        else:
            self.step_function = obstacle_avoidance_acceleration_2d

    def coupling(self, y, yd):
        """Computes coupling term based on current state.

        Parameters
        ----------
        y : array, shape (n_dims,)
            Current position.

        yd : array, shape (n_dims,)
            Current velocity.

        Returns
        -------
        cd : array, shape (n_dims,)
            Velocity. 0 for this coupling term.

        cdd : array, shape (n_dims,)
            Acceleration.
        """
        cdd = self.step_function(
            y, yd, self.obstacle_position, self.gamma, self.beta)
        return np.zeros_like(cdd), cdd


class CouplingTermObstacleAvoidance3D:  # for DMP
    r"""Coupling term for obstacle avoidance in 3D.

    For :class:`DMP` and :class:`DMPWithFinalVelocity`.

    Following [1]_, this coupling term adds an acceleration

    .. math::

        \boldsymbol{C}_t = \gamma \boldsymbol{R} \dot{\boldsymbol{y}}
        \theta \exp(-\beta \theta),

    where

    .. math::

        \theta = \arccos\left( \frac{(\boldsymbol{o} - \boldsymbol{y})^T
        \dot{\boldsymbol{y}}}{|\boldsymbol{o} - \boldsymbol{y}|
        |\dot{\boldsymbol{y}}|} \right)

    and a rotation axis :math:`\boldsymbol{r} =
    (\boldsymbol{o} - \boldsymbol{y}) \times \dot{\boldsymbol{y}}` used to
    compute the rotation matrix :math:`\boldsymbol{R}` that rotates about it
    by 90 degrees for an obstacle at position :math:`\boldsymbol{o}`.

    Intuitively, this coupling term adds a movement perpendicular to the
    current velocity in the plane defined by the direction to the obstacle and
    the current movement direction.

    Parameters
    ----------
    obstacle_position : array, shape (3,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Parameter of obstacle avoidance.

    beta : float, optional (default: 20 / pi)
        Parameter of obstacle avoidance.

    References
    ----------
    .. [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., Schaal, S.
       (2013). Dynamical Movement Primitives: Learning Attractor Models for
       Motor Behaviors. Neural Computation 25 (2), 328-373. DOI:
       10.1162/NECO_a_00393,
       https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
    """
    def __init__(self, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
        self.obstacle_position = obstacle_position
        self.gamma = gamma
        self.beta = beta

    def coupling(self, y, yd):
        """Computes coupling term based on current state.

        Parameters
        ----------
        y : array, shape (n_dims,)
            Current position.

        yd : array, shape (n_dims,)
            Current velocity.

        Returns
        -------
        cd : array, shape (n_dims,)
            Velocity. 0 for this coupling term.

        cdd : array, shape (n_dims,)
            Acceleration.
        """
        cdd = obstacle_avoidance_acceleration_3d(
            y, yd, self.obstacle_position, self.gamma, self.beta)
        return np.zeros_like(cdd), cdd


def obstacle_avoidance_acceleration_3d(
        y, yd, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
    """Compute acceleration for obstacle avoidance in 3D.

    Parameters
    ----------
    y : array, shape (..., 3)
        Current position(s).

    yd : array, shape (..., 3)
        Current velocity / velocities.

    obstacle_position : array, shape (3,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Obstacle avoidance parameter.

    beta : float, optional (default: 20 / pi)
        Obstacle avoidance parameter.

    Returns
    -------
    cdd : array, shape (..., 3)
        Accelerations.
    """
    obstacle_diff = obstacle_position - y
    r = 0.5 * np.pi * pr.norm_vector(np.cross(obstacle_diff, yd))
    R = pr.matrix_from_compact_axis_angle(r)
    theta = np.arccos(
        np.dot(obstacle_diff, yd)
        / (np.linalg.norm(obstacle_diff) * np.linalg.norm(yd) + EPSILON))
    cdd = gamma * np.dot(R, yd) * theta * np.exp(-beta * theta)
    return cdd


class CouplingTermPos1DToPos1D:
    """Couples position components of a 2D DMP with a virtual spring.

    For :class:`DMP` and :class:`DMPWithFinalVelocity`.

    Parameters
    ----------
    desired_distance : float
        Desired distance between components.

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.

    References
    ----------
    .. [1] Gams, A., Nemec, B., Zlajpah, L., Wächter, M., Asfour, T., Ude, A.
       (2013). Modulation of Motor Primitives using Force Feedback: Interaction
       with the Environment and Bimanual Tasks (2013), In 2013 IEEE/RSJ
       International Conference on Intelligent Robots and Systems (pp.
       5629-5635). DOI: 10.1109/IROS.2013.6697172,
       https://h2t.anthropomatik.kit.edu/pdf/Gams2013.pdf
    """
    def __init__(self, desired_distance, lf, k=1.0, c1=100.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y, yd=None):
        da = y[0] - y[1]
        F12 = self.k * (-self.desired_distance - da)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = self.c2 * self.c1 * F12 * self.lf[0]
        C21dot = self.c2 * self.c1 * F21 * self.lf[1]
        return np.array([C12, C21]), np.array([C12dot, C21dot])


class CouplingTermPos3DToPos3D:
    """Couples position components of a 6D DMP with a virtual spring in 3D.

    For :class:`DMP` and :class:`DMPWithFinalVelocity`.

    Parameters
    ----------
    desired_distance : array, shape (3,)
        Desired distance between components.

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.

    References
    ----------
    .. [1] Gams, A., Nemec, B., Zlajpah, L., Wächter, M., Asfour, T., Ude, A.
       (2013). Modulation of Motor Primitives using Force Feedback: Interaction
       with the Environment and Bimanual Tasks (2013), In 2013 IEEE/RSJ
       International Conference on Intelligent Robots and Systems (pp.
       5629-5635). DOI: 10.1109/IROS.2013.6697172,
       https://h2t.anthropomatik.kit.edu/pdf/Gams2013.pdf
    """
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y, yd=None):
        da = y[:3] - y[3:6]
        # Why do we take -self.desired_distance here? Because this allows us
        # to regard the desired distance as the displacement of DMP1 with
        # respect to DMP0.
        F12 = self.k * (-self.desired_distance - da)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = F12 * self.c2 * self.lf[0]
        C21dot = F21 * self.c2 * self.lf[1]
        return np.hstack([C12, C21]), np.hstack([C12dot, C21dot])


class CouplingTermDualCartesianDistance:
    """Couples distance between 3D positions of a dual Cartesian DMP.
        # 双笛卡尔 DMP 3D 位置之间的耦合距离。

    For :class:`DualCartesianDMP`.

    Parameters
    ----------
    desired_distance : float
        Desired distance between components.  # 组件之间的期望距离。

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].
        二进制值，表示将对哪些 DMP 进行调整。
        变量 lf 定义了领导者与追随者的关系。如果 lf[0] = lf[1]，则两个机器人都将调整自己的轨迹，
        以遵循它们之间规定距离 dd 的平均轨迹[...]。反之，如果 lf[0] = 0，lf[1] = 1，
        则只有 DMP1 会改变轨迹，使其与 DMP0 的轨迹相匹配，同样是在距离 dd 处，
        而且只有在学习之后才会改变。反之亦然。领导者与追随者的关系可以由更高级别的规划者决定[...]。

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.  # 耦合位置的虚拟弹簧常数。

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.  # 速度分量和加速度分量中弹簧力的缩放因子。

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.  # 加速度分量中弹簧力的缩放因子。
    """
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        # 初始化类参数
        self.desired_distance = desired_distance  # 期望距离
        self.lf = lf  # 领导-跟随关系
        self.k = k  # 虚拟弹簧弹性系数
        self.c1 = c1  # 缩放因子1
        self.c2 = c2  # 缩放因子2

    def coupling(self, y, yd=None):
        """计算耦合力以及它对速度和加速度的贡献
        Parameters
        ----------
        y : array, shape (14,)
        yd : array, shape (14,1), optional (default: None)

        Returns
        -------
        tupe of two arrays
            - 耦合对速度的影响 (C12, C21)
            - 耦合对加速度的影响 (C12dot, C21dot)
        """

        # 计算两个 DMP 当前的位置差距
        actual_distance = y[:3] - y[7:10]  # DMP0 - DMP1

        # 将目标距离投影到当前方向上，确保方向一致
        desired_distance = (np.abs(self.desired_distance) * actual_distance
                            / np.linalg.norm(actual_distance))  # np.linalg.norm求范数

        # 计算虚拟弹簧力（F12作用在DMP0，F21作用在DMP1）
        F12 = self.k * (desired_distance - actual_distance)
        F21 = -F12  # 力的反作用

        # 计算耦合项对速度的影响（按c1缩放）
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]

        # 计算耦合项对加速度的影响（按c2缩放）
        C12dot = F12 * self.c2 * self.lf[0]
        C21dot = F21 * self.c2 * self.lf[1]

        # 返回速度和加速度耦合修正值
        return (np.hstack([C12, np.zeros(3), C21, np.zeros(3)]),
                np.hstack([C12dot, np.zeros(3), C21dot, np.zeros(3)]))  # np.hstack:行堆叠


class CouplingTermDualCartesianPose:
    """Couples relative poses of dual Cartesian DMP.
    双笛卡尔 DMP 的耦合相对姿势。

    For :class:`DualCartesianDMP`.

    Parameters
    ----------
    desired_distance : array, shape (4, 4)
        Desired distance between components.  # 组件之间的期望距离。

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].

    couple_position : bool, optional (default: True)
        Couple position between components. # 组件之间的耦合位置。

    couple_orientation : bool, optional (default: True)
        Couple orientation between components.  # 组件之间的耦合姿态

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.  # 耦合位置的虚拟弹簧常数。

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.  # 速度分量和加速度分量中弹簧力的缩放因子。

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.
        # 加速度分量中弹簧力的缩放因子。

    verbose : bool, optional (default: False)
        控制调试信息的打印（默认不打印）
    """
    def __init__(self, desired_distance, lf, couple_position=True,
                 couple_orientation=True, k=1.0, c1=1.0, c2=30.0, verbose=0):
        # 初始化目标距离、领导-跟随关系、耦合选项和参数
        self.desired_distance = desired_distance  # 期望距离
        self.lf = lf  # 领导-跟随关系
        self.couple_position = couple_position  # 是否耦合位置
        self.couple_orientation = couple_orientation  # 是否耦合姿态
        self.k = k  # 虚拟弹簧常数
        self.c1 = c1  # 速度缩放因子
        self.c2 = c2  # 加速度缩放因子
        self.verbose = verbose  # 调试信息开关

    def coupling(self, y, yd=None):
        return self.couple_distance(
            y, yd, self.k, self.c1, self.c2, self.lf, self.desired_distance,
            self.couple_position, self.couple_orientation)

    def couple_distance(self, y, yd, k, c1, c2, lf, desired_distance,
                        couple_position, couple_orientation):
        """实现位置和姿态耦合的计算
        Parameters
        ----------
        y : array, shape (14,)
            当前状态

        yd : array, shape (14,1)
            当前速度

        k : float, optional (default: 1)
            虚拟弹簧常数

        c1 : float, optional (default: 100)
            速度缩放因子

        c2 : float, optional (default: 30)
            加速度缩放因子

        lf : array-like, shape (2,)
            定义领导与跟随的关系

        desired_distance : array, shape (4, 4)
            期望距离矩阵

        couple_position : bool, optional (default: True)
            是否耦合位置

        couple_orientation : bool, optional (default: True)
            是否耦合姿态

        Returns
        -------
        耦合修正后的速度和加速度
        """
        # 计算阻尼系数
        damping = 2.0 * np.sqrt(k * c2)

        # 提取 {left} {right} 的当前速度
        vel_left = yd[:6]
        vel_right = yd[6:]

        # 位置和姿态（四元数）计算变换矩阵（4*4齐次矩阵）
        left2base = pt.transform_from_pq(y[:7])  # {left}相对于{base}的齐次变换矩阵
        right2left_pq = self._right2left_pq(y)  # {right}相对于{left}的位姿（位置和姿态（四元数））

        # 分离位置和姿态
        actual_distance_pos = right2left_pq[:3]  # 实际相对位置
        actual_distance_rot = right2left_pq[3:]  # 实际相对姿态

        # 从期望距离齐次矩阵提取位置和姿态（四元数）
        desired_distance = pt.pq_from_transform(desired_distance)
        desired_distance_pos = desired_distance[:3]
        desired_distance_rot = desired_distance[3:]

        # 调试信息
        if self.verbose:
            print("Desired vs. actual:")
            print(np.round(desired_distance, 2))
            print(np.round(right2left_pq, 2))

        # 位置耦合 -------
        error_pos = desired_distance_pos - actual_distance_pos  # 位置误差
        F12_pos = -k * error_pos  # 作用于组件1的弹簧力
        F21_pos = k * error_pos  # 作用于组件2的弹簧力

        # 将弹簧力转换到全局坐标系
        F12_pos = pt.transform(left2base, pt.vector_to_direction(F12_pos))[:3]
        F21_pos = pt.transform(left2base, pt.vector_to_direction(F21_pos))[:3]
        # transform:Bp=BAT*Ap PB=transform(A2B, PA)==>F12_pos=left2base*pt.vector_to_direction(F12_pos)
        # vector_to_direction:将三维矢量转换为方向 (x,y,z)->[x,y,z,0]（齐次坐标表达）

        # 速度修正项
        C12_pos = lf[0] * c1 * F12_pos
        C21_pos = lf[1] * c1 * F21_pos

        # 加速度修正项
        C12dot_pos = lf[0] * (c2 * F12_pos - damping * vel_left[:3])
        C21dot_pos = lf[1] * (c2 * F21_pos - damping * vel_right[:3])

        # 如果不耦合位置，则清零相关修正项
        if not couple_position:
            C12_pos *= 0
            C21_pos *= 0
            C12dot_pos *= 0
            C21dot_pos *= 0

        # 姿态耦合 --------（同上）
        error_rot = pr.compact_axis_angle_from_quaternion(
            pr.concatenate_quaternions(desired_distance_rot,
                                       pr.q_conj(actual_distance_rot)))  # 姿态误差
        F12_rot = -k * error_rot
        F21_rot = k * error_rot

        F12_rot = pt.transform(left2base, pt.vector_to_direction(F12_rot))[:3]
        F21_rot = pt.transform(left2base, pt.vector_to_direction(F21_rot))[:3]

        C12_rot = lf[0] * c1 * F12_rot
        C21_rot = lf[1] * c1 * F21_rot

        C12dot_rot = lf[0] * (c2 * F12_rot - damping * vel_left[3:])
        C21dot_rot = lf[1] * (c2 * F21_rot - damping * vel_right[3:])

        if not couple_orientation:
            C12_rot *= 0
            C21_rot *= 0
            C12dot_rot *= 0
            C21dot_rot *= 0

        # 返回耦合后的修正值
        return (np.hstack([C12_pos, C12_rot, C21_pos, C21_rot]),
                np.hstack([C12dot_pos, C12dot_rot, C21dot_pos, C21dot_rot]))

    def _right2left_pq(self, y):
        """计算{right}相对于{left}的位姿（位置和姿态（四元数））

        Parameters
        ----------
        y : array, shape (14,)
            当前状态

        Returns
        -------
        right2left_pq : array, shape (7,)
            包含相对位置和姿态（四元数）
        """
        left2base = pt.transform_from_pq(y[:7])     # {left}相对于{base}的的齐次变换矩阵
        right2base = pt.transform_from_pq(y[7:])    # {right}相对于{base}的齐次变换矩阵
        base2left = pt.invert_transform(left2base)  # {base}相对于{left}的齐次变换矩阵
        right2left = pt.concat(right2base, base2left)   # {right}相对于{left}的齐次变换矩阵
        right2left_pq = pt.pq_from_transform(right2left)# {right}相对于{left}的位姿（位置和四元数姿态）
        return right2left_pq


class CouplingTermDualCartesianTrajectory(CouplingTermDualCartesianPose):
    """Couples relative pose in dual Cartesian DMP with a given trajectory.
    将双笛卡尔 DMP 中的相对姿态与给定轨迹进行耦合。

    For :class:`DualCartesianDMP`.

    Parameters
    ----------
    offset : array, shape (7,)
        Offset for desired distance between components as position and
        quaternion.
        作为位置和四元数的组件之间所需的距离偏移。

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].
        二进制值，表示将对哪些 DMP 进行调整。
        变量 lf 定义了leader与follower的关系。如果 lf[0] = lf[1]，则两个机器人都将调整自己的轨迹，
        以遵循它们之间规定距离 dd 的平均轨迹[...]。反之，如果 lf[0] = 0，lf[1] = 1，
        则只有 DMP1 会改变轨迹，使其与 DMP0 的轨迹相匹配，同样是在距离 dd 处，也只有
        在学习之后才会改变。反之亦然。领导者与追随者的关系可以由更高级别的规划者决定[...]。

    couple_position : bool, optional (default: True)
        Couple position between components.
        组件之间的耦合位置。

    couple_orientation : bool, optional (default: True)
        Couple orientation between components.
        组件之间的耦合姿态

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.
        耦合位置的虚拟弹簧常数。

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.
        速度分量和加速度分量中弹簧力的缩放因子。

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.
        加速度分量中弹簧力的缩放因子。
    """
    def __init__(self, offset, lf, dt, couple_position=True,
                 couple_orientation=True, k=1.0, c1=1.0, c2=30.0, verbose=1):
        # 初始化类成员变量
        self.offset = offset  # 偏移量
        self.lf = lf  # 领导-跟随关系
        self.dt = dt  # 时间不长
        self.couple_position = couple_position  # 是否耦合位置
        self.couple_orientation = couple_orientation  # 是否耦合姿态
        self.k = k  # 虚拟弹簧常数
        self.c1 = c1  # 速度缩放因子
        self.c2 = c2  # 加速度缩放因子
        self.verbose = verbose  # 调试信息开关

    def imitate(self, T, Y):
        """模仿轨迹，生成期望的相对位姿轨迹

        Parameters
        ----------
        T : array, shape
            时间序列

        Y : array, shape( ,14)
            笛卡尔轨迹，每一行包含两个组件的位置和方向

        功能
        -------

        """
        distance = np.empty((len(Y), 7))  # 初始化相对位姿

        # 计算每个时间点{right}相对于{left}的位姿（位置和姿态（四元数））
        for t in range(len(Y)):
            distance[t] = self._right2left_pq(Y[t])

        # 为每个维度生成插值函数
        self.desired_distance_per_dimension = [
            interp1d(T, distance[:, d], bounds_error=False,
                     fill_value="extrapolate")
            for d in range(distance.shape[1])
        ]
        self.t = 0.0  # 初始化时间计数器

    def coupling(self, y, yd=None):
        """计算耦合修正值
        Parameters
        ----------
        y : array, shape( ,14)
            当前状态（两个组件的位姿）

        yd : array, shape( ,14) , optional (default: None)
            当前速度（默认为None）

        Returns
        -------
        耦合修正的速度和加速度
        """
        # 根据当前时间`self.t`，从插值函数中计算目标相对姿态
        desired_distance = np.empty(len(self.desired_distance_per_dimension))  # 初始化self.t时间的相对目标姿态
        for d in range(len(desired_distance)):
            desired_distance[d] = self.desired_distance_per_dimension[d](self.t)
        desired_distance += self.offset  # 添加偏移量
        self.t += self.dt   # 更新时间计数器

        # 调用父类方法`couple_distance`计算耦合修正
        return self.couple_distance(
            y, yd, self.k, self.c1, self.c2, self.lf,
            pt.transform_from_pq(desired_distance), self.couple_position,
            self.couple_orientation)
