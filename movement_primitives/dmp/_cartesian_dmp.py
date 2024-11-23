import numpy as np
import pytransform3d.rotations as pr
from ..utils import ensure_1d_array
from ._base import DMPBase
from ._forcing_term import ForcingTerm
from ._canonical_system import canonical_system_alpha
from ._dmp import (dmp_open_loop, dmp_imitate, ridge_regression,
                   DMP_STEP_FUNCTIONS, DEFAULT_DMP_STEP_FUNCTION, phase)

def dmp_step_quaternion_python(
        last_t, t,
        current_y, current_yd,
        goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd,
        goal_t, start_t, alpha_y, beta_y,
        forcing_term,
        coupling_term=None,
        coupling_term_precomputed=None,
        int_dt=0.001,
        smooth_scaling=False):
    """Integrate quaternion DMP for one step with Euler integration.  # 用欧拉积分法对四元数 DMP 进行一步积分。

    Parameters
    ----------
    last_t : float
        Time at last step.

    t : float
        Time at current step.

    current_y : array, shape (7,)
        Current position. Will be modified.

    current_yd : array, shape (6,)
        Current velocity. Will be modified.

    goal_y : array, shape (7,)
        Goal position.

    goal_yd : array, shape (6,)
        Goal velocity.

    goal_ydd : array, shape (6,)
        Goal acceleration.

    start_y : array, shape (7,)
        Start position.

    start_yd : array, shape (6,)
        Start velocity.

    start_ydd : array, shape (6,)
        Start acceleration.

    goal_t : float
        Time at the end.

    start_t : float
        Time at the start.

    alpha_y : array, shape (6,)
        Constant in transformation system.

    beta_y : array, shape (6,)
        Constant in transformation system.

    forcing_term : ForcingTerm
        Forcing term.

    coupling_term : CouplingTerm, optional (default: None)
        Coupling term. Must have a function coupling(y, yd) that returns
        additional velocity and acceleration.

    coupling_term_precomputed : tuple
        A precomputed coupling term, i.e., additional velocity and
        acceleration.

    int_dt : float, optional (default: 0.001)
        Time delta used internally for integration.

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.
        通过对新旧轨迹缩放比例进行内插，避免在 DMP 执行之初目标发生变化和轨迹缩放时出现跳转。

    Raises
    ------
    ValueError
        If goal time is before start time.
    """
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    if t <= start_t:
        return np.copy(start_y), np.copy(start_yd), np.copy(start_ydd)

    execution_time = goal_t - start_t

    current_ydd = np.empty_like(current_yd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd, cdd = coupling_term.coupling(current_y, current_yd)
        else:
            cd, cdd = np.zeros(3), np.zeros(3)
        if coupling_term_precomputed is not None:
            cd += coupling_term_precomputed[0]
            cdd += coupling_term_precomputed[1]

        z = forcing_term.phase(current_t)
        f = forcing_term.forcing_term(z).squeeze()

        if smooth_scaling:
            goal_y_minus_start_y = pr.compact_axis_angle_from_quaternion(
                pr.concatenate_quaternions(goal_y, pr.q_conj(start_y)))
            smoothing = beta_y * z * goal_y_minus_start_y
        else:
            smoothing = 0.0

        current_ydd[:] = (
            alpha_y * (
                beta_y * pr.compact_axis_angle_from_quaternion(
                    pr.concatenate_quaternions(goal_y, pr.q_conj(current_y))
                )
                - execution_time * current_yd
                - smoothing
            )
            + f
            + cdd
        ) / execution_time ** 2
        current_yd += dt * current_ydd + cd / execution_time
        current_y[:] = pr.concatenate_quaternions(
            pr.quaternion_from_compact_axis_angle(dt * current_yd), current_y)


CARTESIAN_DMP_STEP_FUNCTIONS = {
    "python": dmp_step_quaternion_python
}


try:
    from ..dmp_fast import dmp_step_quaternion
    CARTESIAN_DMP_STEP_FUNCTIONS["cython"] = dmp_step_quaternion
    DEFAULT_CARTESIAN_DMP_STEP_FUNCTION = "cython"
except ImportError:
    DEFAULT_CARTESIAN_DMP_STEP_FUNCTION = "python"


class CartesianDMP(DMPBase):
    r"""Cartesian dynamical movement primitive.

    The Cartesian DMP handles orientation and position separately. The
    orientation is represented by a quaternion.

    While the dimension of the state space is 7, the dimension of the
    velocity, acceleration, and forcing term is 6.

    Equation of transformation system for the orientation (according to [1]_,
    Eq. 16):
    笛卡尔 DMP 分别处理方向和位置。方向由四元数表示。

    状态空间的维数为 7，而速度、加速度和力项的维数为 6。

    方位变换系统方程（根据 [1]_，公式 16）：

    .. math::

        \ddot{y} = (\alpha_y (\beta_y (g - y) - \tau \dot{y}) + f(z) + C_t) / \tau^2

    Note that in this case :math:`y` is a quaternion in this case,
    :math:`g - y` the quaternion difference (expressed as rotation vector),
    :math:`\dot{y}` is the angular velocity, and :math:`\ddot{y}` the
    angular acceleration.

    With smooth scaling (according to [2]_):

    .. math::

        \ddot{y} = (\alpha_y (\beta_y (g - y) - \tau \dot{y}
        - \underline{\beta_y (g - y_0) z}) + f(z) + C_t) / \tau^2

    The position is handled in the same way, just like in the original
    :class:`DMP`.

    Parameters
    ----------
    execution_time : float, optional (default: 1)
        Execution time of the DMP.  # DMP 的执行时间。

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.  # DMP 步之间的时间差。

    n_weights_per_dim : int, optional (default: 10)
        Number of weights of the function approximator per dimension.  # 每个维度的函数逼近器权重数。

    int_dt : float, optional (default: 0.001)
        Time difference for Euler integration.  # 欧拉积分的时间差

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.
        通过对新旧轨迹缩放比例进行内插，避免在 DMP 执行之初目标发生变化和轨迹缩放时出现跳转。

    alpha_y : float or array-like, shape (6,), optional (default: 25.0)
        Parameter of the transformation system.  # 变化系统的参数

    beta_y : float or array-like, shape (6,), optional (default: 6.25)
        Parameter of the transformation system.  # 变换系统的参数

    Attributes
    ----------
    execution_time_ : float
        Execution time of the DMP.  # DMP的执行时间

    dt_ : float
        Time difference between DMP steps. This value can be changed to adapt
        the frequency.  # DMP 步之间的时间差。该值可以更改，以调整频率。

    References
    ----------
    .. [1] Ude, A., Nemec, B., Petric, T., Murimoto, J. (2014).
       Orientation in Cartesian space dynamic movement primitives.
       In IEEE International Conference on Robotics and Automation (ICRA)
       (pp. 2997-3004). DOI: 10.1109/ICRA.2014.6907291,
       https://acat-project.eu/modules/BibtexModule/uploads/PDF/udenemecpetric2014.pdf

    .. [2] Pastor, P., Hoffmann, H., Asfour, T., Schaal, S. (2009). Learning
       and Generalization of Motor Skills by Learning from Demonstration.
       In 2009 IEEE International Conference on Robotics and Automation,
       (pp. 763-768). DOI: 10.1109/ROBOT.2009.5152385,
       https://h2t.iar.kit.edu/pdf/Pastor2009.pdf
    """
    def __init__(
            self, execution_time=1.0, dt=0.01, n_weights_per_dim=10,
            int_dt=0.001, smooth_scaling=False, alpha_y=25.0, beta_y=6.25):
        # 调用父类 DMPBase 构造器，指定状态空间和速度空间维度
        super(CartesianDMP, self).__init__(7, 6)

        # 初始化类属性
        self._execution_time = execution_time  # DMP 的执行时间
        self.dt_ = dt  # DMP 步之间的时间差
        self.n_weights_per_dim = n_weights_per_dim  # 每个维度的函数逼近器权重数
        self.int_dt = int_dt  # 欧拉积分的时间差
        self.smooth_scaling = smooth_scaling  # 是否启用平滑缩放

        # 初始化力项 (forcing term)
        self._init_forcing_term()

        # 确保 alpha_y 和 beta_y 是长度为 6 的一维数组
        self.alpha_y = ensure_1d_array(alpha_y, 6, "alpha_y")  # ensure_1d_array:自编函数
        self.beta_y = ensure_1d_array(beta_y, 6, "beta_y")

    def _init_forcing_term(self):
        """
        初始化力项 (forcing term)

        根据笛卡尔空间的位置和方向分别确定力项。
        使用 canonical_system_alpha 函数生成力项相关的 alpha 参数

        - 力项分为两个部分：位置力项和旋转力项。
        - 每个力项使用 ForcingTerm 类生成，其核心参数包括维度、权重数、执行时间等。

        """
        alpha_z = canonical_system_alpha(0.01, self.execution_time_, 0.0)  # 计算正则系统的alpha参数，适应DMP的执行时间
        self.forcing_term_pos = ForcingTerm(
            3, self.n_weights_per_dim, self.execution_time_, 0.0, 0.8,
            alpha_z)  # 初始化位置力项(3维)
        self.forcing_term_rot = ForcingTerm(
            3, self.n_weights_per_dim, self.execution_time_, 0.0, 0.8,
            alpha_z)

    def get_execution_time_(self):
        """
        获取当前DMP执行时间

        Returns
        -------
        _execution_time : float
            当前的执行时间
        """
        return self._execution_time

    def set_execution_time_(self, execution_time):
        """
        更新DMP的执行时间，并重新初始化力项

        更新执行时间后，为了保持一致性，需要重新初始化力项对象。
        在初始化后，保留先前的力项权重，避免重新生成权重数据。

        Parameters
        ----------
        execution_time : float
            新的执行时间
        """
        self._execution_time = execution_time  # 更新执行时间

        # 暂存当前的位置和方向力项的权重
        weights_pos = self.forcing_term_pos.weights_
        weights_rot = self.forcing_term_rot.weights_
        self._init_forcing_term()
        self.forcing_term_pos.weights_ = weights_pos
        self.forcing_term_rot.weights_ = weights_rot

    execution_time_ = property(get_execution_time_, set_execution_time_)
   # 属性`exection_time_` 使用`property`装饰器实现动态管理

    def step(self, last_y, last_yd, coupling_term=None,
             step_function=DMP_STEP_FUNCTIONS[DEFAULT_DMP_STEP_FUNCTION],
             quaternion_step_function=CARTESIAN_DMP_STEP_FUNCTIONS[
                 DEFAULT_CARTESIAN_DMP_STEP_FUNCTION]):
        """DMP step.  # DMP 单步计算方法。

        每次调用此方法会基于上一步的状态和速度，计算当前时刻的状态和速度。
        该方法分别处理位置和方向（四元数）的积分更新

        Parameters
        ----------
        last_y : array, shape (7,)
            Last state.  # 上一次的状态。

        last_yd : array, shape (6,)
            Last time derivative of state (velocity).  # 状态（速度）的上一次导数。

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.  # 将添加到速度中的耦合项。

        step_function : callable, optional (default: RK4)
            DMP integration function.  # DMP积分函数

        quaternion_step_function : callable, optional (default: cython code if available)
            DMP integration function.  # DMP积分函数（四元数）

        Returns
        -------
        y : array, shape (14,)
            Next state.  # 下一次状态

        yd : array, shape (12,)
            Next time derivative of state (velocity).  # 状态（速度）的下一次导数。
        """
        assert len(last_y) == 7  # assert:断言，用于判断一个表达式，在表达式条件为false的时候触发异常
        assert len(last_yd) == 6

        # 更新时间戳：记录上一次时间并推进当前时间
        self.last_t = self.t
        self.t += self.dt_

        # TODO tracking error

        # 将上一次的状态和速度赋给当前变量
        self.current_y[:], self.current_yd[:] = last_y, last_yd

        # 使用积分函数更新位置部分
        step_function(
            self.last_t, self.t,
            self.current_y[:3], self.current_yd[:3],
            self.goal_y[:3], self.goal_yd[:3], self.goal_ydd[:3],
            self.start_y[:3], self.start_yd[:3], self.start_ydd[:3],
            self.execution_time_, 0.0,
            self.alpha_y[:3], self.beta_y[:3],
            self.forcing_term_pos,  # 位置力项
            coupling_term=coupling_term,  # 耦合项
            int_dt=self.int_dt,  # 积分时间步长
            smooth_scaling=self.smooth_scaling)

        # 使用step函数更新姿态部分
        quaternion_step_function(
            self.last_t, self.t,
            self.current_y[3:], self.current_yd[3:],
            self.goal_y[3:], self.goal_yd[3:], self.goal_ydd[3:],
            self.start_y[3:], self.start_yd[3:], self.start_ydd[3:],
            self.execution_time_, 0.0,
            self.alpha_y[3:], self.beta_y[3:],
            self.forcing_term_rot,  # 姿态力项
            coupling_term=coupling_term,
            int_dt=self.int_dt,
            smooth_scaling=self.smooth_scaling)
        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=None, coupling_term=None,
                  step_function=DEFAULT_DMP_STEP_FUNCTION,
                  quaternion_step_function=DEFAULT_CARTESIAN_DMP_STEP_FUNCTION):
        """Run DMP open loop.  # 开环运行 DMP。

        Parameters
        ----------
        run_t : float, optional (default: execution_time)
            Run time of DMP. Can be shorter or longer than execution_time.
            # DMP 的运行时间，可以比execution_time更长或更短 。

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.  # 将添加到速度中的耦合项。

        step_function : str, optional (default: 'rk4')
            DMP integration function. Possible options: 'rk4', 'euler',
            'euler-cython', 'rk4-cython'.
            # DMP 积分函数。可选项:"rk4“、”euler“、”euler-cython“、”rk4-cython"。

        quaternion_step_function : str, optional (default: 'cython' if available)
            DMP integration function. Possible options: 'python', 'cython'.
            # DMP 积分函数。可选项：“python”、“cython”。

        Returns
        -------
        T : array, shape (n_steps,)
            Time for each step.  # 每个时间步的时间序列。

        Y : array, shape (n_steps, 7)
            State at each step.  # 每一步的状态。
        """
        try:
            step_function = DMP_STEP_FUNCTIONS[step_function]
        except KeyError:
            raise ValueError(
                f"Step function must be in "
                f"{DMP_STEP_FUNCTIONS.keys()}.")

        # 计算位置部分的轨迹(Yp)
        T, Yp = dmp_open_loop(
            self.execution_time_, 0.0, self.dt_,
            self.start_y[:3], self.goal_y[:3],
            self.alpha_y[:3], self.beta_y[:3],
            self.forcing_term_pos,
            coupling_term,
            run_t, self.int_dt,
            step_function=step_function,
            smooth_scaling=self.smooth_scaling)

        try:
            quaternion_step_function = CARTESIAN_DMP_STEP_FUNCTIONS[
                quaternion_step_function]
        except KeyError:
            raise ValueError(
                f"Step function must be in "
                f"{CARTESIAN_DMP_STEP_FUNCTIONS.keys()}.")

        # 计算姿态部分的轨迹(Yr)
        _, Yr = dmp_open_loop_quaternion(
            self.execution_time_, 0.0, self.dt_,
            self.start_y[3:], self.goal_y[3:],
            self.alpha_y[3:], self.beta_y[3:],
            self.forcing_term_rot,
            coupling_term,
            run_t, self.int_dt,
            quaternion_step_function,
            self.smooth_scaling)
        return T, np.hstack((Yp, Yr))  # 水平堆叠序列中的数组（列方向）

    def imitate(self, T, Y, regularization_coefficient=0.0,
                allow_final_velocity=False):
        """Imitate demonstration.  # 模仿示范。

        Target forces for the forcing term are computed for the positions
        in a similar way as in :func:`DMP.imitate`. For the orientations
        we adapt this to handle quaternions adequately.
        力项的目标力是以类似于 :func:`DMP.imitate` 的方式计算位置的。
        对于方向，我们将对其进行调整，以充分处理四元数。

        Parameters
        ----------
        T : array, shape (n_steps,)
            Time for each step.  # 每一步的时间。

        Y : array, shape (n_steps, 7)
            State at each step.  # 每一步的状态。

        regularization_coefficient : float, optional (default: 0)
            Regularization coefficient for regression.  # 回归的正则化系数。

        allow_final_velocity : bool, optional (default: False)
            Allow a final velocity.  # 允许最终速度。
        """
        self.forcing_term_pos.weights_[:, :] = dmp_imitate(
            T, Y[:, :3],                                # 时间序列和位置轨迹
            n_weights_per_dim=self.n_weights_per_dim,   # 每个维度的权重数
            regularization_coefficient=regularization_coefficient,  # 正则化系数
            alpha_y=self.alpha_y[:3], beta_y=self.beta_y[:3],  # 位置的alpha_y和beta_y系数
            overlap=self.forcing_term_pos.overlap,      # 高斯核的重叠参数
            alpha_z=self.forcing_term_pos.alpha_z,      # 正则系统的alpha_z系数
            allow_final_velocity=allow_final_velocity,  # 是否允许最终速度
            smooth_scaling=self.smooth_scaling          # 平滑缩放，避免轨迹冲突
        )[0]  # 返回结果weights部分

        # 使用示教数据计算姿态部分（四元数）的力项权重
        self.forcing_term_rot.weights_[:, :] = dmp_quaternion_imitation(
            T, Y[:, 3:],                                # 时间序列和方向轨迹（四元数）
            n_weights_per_dim=self.n_weights_per_dim,   # 每个维度的权重数
            regularization_coefficient=regularization_coefficient,  # 正则化系数
            alpha_y=self.alpha_y[3:], beta_y=self.beta_y[3:],  # 姿态的alpha_y和beta_y系数
            overlap=self.forcing_term_rot.overlap,      # 高斯核的重叠参数
            alpha_z=self.forcing_term_rot.alpha_z,      # 正则系统的alpha_z系数
            allow_final_velocity=allow_final_velocity,  # 是否允许最终速度
            smooth_scaling=self.smooth_scaling          # 平滑缩放，避免轨迹冲突
        )[0]  # 返回结果weights部分

        # 配置起始状态和目标状态
        self.configure(start_y=Y[0], goal_y=Y[-1])

    def get_weights(self):
        """Get weight vector of DMP.  # 获取 DMP 的权重向量。

        Returns
        -------
        weights : array, shape (6 * n_weights_per_dim,)
            Current weights of the DMP.  # DMP 的当前权重。
        """
        # 将位置部分和姿态部分的权重展平并拼接成一个长向量
        return np.concatenate((self.forcing_term_pos.weights_.ravel(),
                               self.forcing_term_rot.weights_.ravel()))

    def set_weights(self, weights):
        """Set weight vector of DMP.  # 设置 DMP 的权重向量。

        Parameters
        ----------
        weights : array, shape (6 * n_weights_per_dim,)
            New weights of the DMP.  # DMP 的新权重。
        """
        n_pos_weights = self.forcing_term_pos.weights_.size  # 获取位置部分权重的大小
        self.forcing_term_pos.weights_[:, :] = weights[:n_pos_weights].reshape(
            -1, self.n_weights_per_dim)  # 将位置部分的权重赋值给forcing_term_pos
        self.forcing_term_rot.weights_[:, :] = weights[n_pos_weights:].reshape(
            -1, self.n_weights_per_dim)  # 将姿态部分的权重赋值给forcing_term_rot


def dmp_quaternion_imitation(
        T, Y, n_weights_per_dim, regularization_coefficient, alpha_y, beta_y,
        overlap, alpha_z, allow_final_velocity, smooth_scaling=False):
    """Compute weights and metaparameters of quaternion DMP.  # 计算四元数 DMP 的权重和元参数。

    Parameters
    ----------
    T : array, shape (n_steps,)
        Time of each step.  # 每一步的时间。

    Y : array, shape (n_steps, 4)
        Orientation at each step.  # 每个步的姿态

    n_weights_per_dim : int
        Number of weights per dimension.  # 每个维度的权重数量

    regularization_coefficient : float, optional (default: 0)
        Regularization coefficient for regression.  # 回归正则化系数。

    alpha_y : array, shape (3,)
        Parameter of the transformation system.  # 变换系统的参数

    beta_y : array, shape (3,)
        Parameter of the transformation system.  # 变换系统的参数

    overlap : float
        At which value should radial basis functions of the forcing term
        overlap?  # 力项的径向基函数应在哪个值上重叠？

    alpha_z : float
        Parameter of the canonical system.  # 正则系统的参数

    allow_final_velocity : bool
        Whether a final velocity is allowed. Will be set to 0 otherwise.
        是否允许最终速度。否则将设为 0。

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.
        通过对新旧轨迹缩放比例进行内插，避免在 DMP 执行之初目标发生变化和轨迹缩放时出现跳转。

    Returns
    -------
    weights : array, shape (3, n_weights_per_dim)
        Weights of the forcing term.  # 力项的权重

    start_y : array, shape (4,)
        Start orientation.  # 起始姿态

    start_yd : array, shape (3,)
        Start velocity.  # 起始速度

    start_ydd : array, shape (3,)
        Start acceleration.  # 起始加速度

    goal_y : array, shape (4,)
        Final orientation.  # 最终的姿态

    goal_yd : array, shape (3,)
        Final velocity.  # 最终速度

    goal_ydd : array, shape (3,)
        Final acceleration.  # 最终加速度
    """
    if regularization_coefficient < 0.0:
        raise ValueError("Regularization coefficient must be >= 0!")

    # 创建力项实例（指定基函数的数量和参数）
    forcing_term = ForcingTerm(
        3, n_weights_per_dim, T[-1], T[0], overlap, alpha_z)

    # 计算目标力以及起点、目标点的方向、速度和加速度
    F, start_y, start_yd, start_ydd, goal_y, goal_yd, goal_ydd = \
        determine_forces_quaternion(
            T, Y, alpha_y, beta_y, alpha_z, allow_final_velocity,
            smooth_scaling)  # n_steps x n_dims

    # 生成设计矩阵（径向基函数基于时间步的值）
    X = forcing_term.design_matrix(T)  # n_weights_per_dim x n_steps

    # 使用岭回归计算权重，返回权重和元参数
    return (ridge_regression(X, F, regularization_coefficient),
            start_y, start_yd, start_ydd, goal_y, goal_yd, goal_ydd)


def determine_forces_quaternion(
        T, Y, alpha_y, beta_y, alpha_z, allow_final_velocity,
        smooth_scaling=False):
    """Determine forces that the forcing term should generate.  # 确定力项应产生的力

    Parameters
    ----------
    T : array, shape (n_steps,)
        Time of each step.  # 每一步的时间。

    Y : array, shape (n_steps, n_dims)
        Position at each step.  # 每一步的位置。

    alpha_y : array, shape (6,)
        Parameter of the transformation system.  # 变换系统的参数

    beta_y : array, shape (6,)
        Parameter of the transformation system.  # 变换系统的参数

    alpha_z : float
        Parameter of the canonical system.  # 正则系统的参数

    allow_final_velocity : bool
        Whether a final velocity is allowed. Will be set to 0 otherwise.
        是否允许最终速度。否则将设为 0。

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.
        通过对新旧轨迹缩放比例进行内插，避免在 DMP 执行之初目标发生变化和轨迹缩放时出现跳转。

    Returns
    -------
    F : array, shape (n_steps, n_dims)
        Forces.  # 力

    start_y : array, shape (4,)
        Start orientation.  # 起始姿态

    start_yd : array, shape (3,)
        Start velocity.  # 起始速度

    start_ydd : array, shape (3,)
        Start acceleration.  # 起始加速度

    goal_y : array, shape (4,)
        Final orientation.  # 最终姿态

    goal_yd : array, shape (3,)
        Final velocity.  # 最终速度

    goal_ydd : array, shape (3,)
        Final acceleration.  # 最终加速度
    """
    n_dims = 3  # 四元数产生的向量是3维

    # 计算时间梯度，用于后续速度和加速度计算
    DT = np.gradient(T)

    # 使用四元数求导计算速度
    Yd = pr.quaternion_gradient(Y) / DT[:, np.newaxis]
    if not allow_final_velocity:
        Yd[-1, :] = 0.0  # 如果不允许终点速度，将其置为0

    # 使用四元数计算加速度
    Ydd = np.empty_like(Yd)
    for d in range(n_dims):
        Ydd[:, d] = np.gradient(Yd[:, d]) / DT
    Ydd[-1, :] = 0.0  # 加速度置为0

    # 计算执行时间、起始姿态和目标姿态
    execution_time = T[-1] - T[0]
    goal_y = Y[-1]
    start_y = Y[0]

    # 目标姿态和起始姿态的误差
    goal_y_minus_start_y = pr.compact_axis_angle_from_quaternion(
        pr.concatenate_quaternions(goal_y, pr.q_conj(start_y)))
    # compact_axis_angle_from_quaternion:根据四元数计算紧凑轴角 (w, x, y, z)-->angle * (x, y, z)
    # q_conj:四元数的共轭 concatenate_quaternions:连接两个四元数 q=q1*q2

    # 计算相位变量（正则系统的状态）
    S = phase(T, alpha_z, T[-1], T[0])  # x_j = x(t_j)=exp(-alpha_x/tau*t_j)

    # 初始化力项矩阵
    F = np.empty((len(T), n_dims))

    # 逐步计算每个时间点的目标力
    for t in range(len(T)):
        if smooth_scaling:
            smoothing = beta_y * S[t] * goal_y_minus_start_y
        else:
            smoothing = 0.0
        F[t, :] = execution_time ** 2 * Ydd[t] - alpha_y * (
            beta_y * pr.compact_axis_angle_from_quaternion(
                pr.concatenate_quaternions(goal_y, pr.q_conj(Y[t])))
            - execution_time * Yd[t]
            - smoothing
        )

    # 返回力项的力和起点、目标点的状态
    return F, Y[0], Yd[0], Ydd[0], Y[-1], Yd[-1], Ydd[-1]


def dmp_open_loop_quaternion(
        goal_t, start_t, dt, start_y, goal_y, alpha_y, beta_y, forcing_term,
        coupling_term=None, run_t=None, int_dt=0.001,
        quaternion_step_function=CARTESIAN_DMP_STEP_FUNCTIONS[
            DEFAULT_CARTESIAN_DMP_STEP_FUNCTION],
        smooth_scaling=False):
    """Run Cartesian DMP without external feedback.  # 在没有外部反馈的情况下运行笛卡尔 DMP。

    Parameters
    ----------
    goal_t : float
        Time at the end.  # 结束时间。

    start_t : float
        Time at the start.  # 起始时间

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.  # DMP步的时间差

    start_y : array, shape (7,)
        Start position.  # 起始位置

    goal_y : array, shape (7,)
        Goal position.  # 终点位置

    alpha_y : array, shape (6,)
        Constant in transformation system.  # 变换系统中的常数

    beta_y : array, shape (6,)
        Constant in transformation system.  # 变换系统中的常数

    forcing_term : ForcingTerm
        Forcing term.  # 力项

    coupling_term : CouplingTerm, optional (default: None)
        Coupling term. Must have a function coupling(y, yd) that returns
        additional velocity and acceleration.
        耦合项。必须有一个返回附加速度和加速度的函数 coupling(y,yd)。

    run_t : float, optional (default: goal_t)
        Time at which the DMP will be stopped.
        DMP 停止运行的时间。

    int_dt : float, optional (default: 0.001)
        Time delta used internally for integration.  # 内部用于积分的时间差

    quaternion_step_function : callable, optional (default: cython code if available)
        DMP integration function.  # DMP积分函数

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.
        通过对新旧轨迹缩放比例进行内插，避免在 DMP 执行之初目标发生变化和轨迹缩放时出现跳转。

    Returns
    -------
    T : array, shape (n_steps,)
        Times.  # 时间

    Y : array, shape (n_steps, 4)
        Orientations.  # 姿态
    """
    # 初始化目标点和起始点的速度和加速度（默认为0）
    goal_yd = np.zeros(3)
    goal_ydd = np.zeros(3)
    start_yd = np.zeros(3)
    start_ydd = np.zeros(3)

    # 如果未指定运行时间，默认运行到目标时间
    if run_t is None:
        run_t = goal_t

    # 初始化当前状态
    current_y = np.copy(start_y)  # 当前姿态（四元数）
    current_yd = np.copy(start_yd)  # 当前速度

    # 生成时间序列和初始化姿态轨迹矩阵
    T = np.arange(start_t, run_t + dt, dt)
    Y = np.empty((len(T), len(current_y)))
    Y[0] = current_y

    # 开环运行DMP
    for i in range(1, len(T)):
        # 调用步进函数，更新状态
        quaternion_step_function(
            T[i - 1], T[i], current_y, current_yd,
            goal_y=goal_y, goal_yd=goal_yd, goal_ydd=goal_ydd,
            start_y=start_y, start_yd=start_yd, start_ydd=start_ydd,
            goal_t=goal_t, start_t=start_t,
            alpha_y=alpha_y, beta_y=beta_y,
            forcing_term=forcing_term, coupling_term=coupling_term,
            int_dt=int_dt, smooth_scaling=smooth_scaling)
        Y[i] = current_y  # 保存当前姿态到轨迹

    # 返回时间序列和姿态轨迹
    return T, Y
