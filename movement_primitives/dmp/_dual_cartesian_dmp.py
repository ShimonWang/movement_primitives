import numpy as np
import pytransform3d.rotations as pr  # 用于四元数和旋转相关计算
from ._base import DMPBase, WeightParametersMixin  # DMP的基类与权重参数混合类
from ..utils import ensure_1d_array  # 工具函数，用于保证数组是一维
from ._canonical_system import canonical_system_alpha  # 正则系统参数
from ._forcing_term import ForcingTerm, phase  # 力项类与相位函数
from ._dmp import dmp_imitate  # DMP模仿学习函数
from ._cartesian_dmp import dmp_quaternion_imitation  # 四元数模仿学习函数


# 笛卡尔位置和速度的索引，分别适用于双臂
pps = [0, 1, 2, 7, 8, 9]  # 平移(position)索引
pvs = [0, 1, 2, 6, 7, 8]  # 速度(velocity)索引


def dmp_step_dual_cartesian_python(
        last_t, t,
        current_y, current_yd,
        goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd,
        goal_t, start_t, alpha_y, beta_y,
        forcing_term, coupling_term=None, int_dt=0.001,
        p_gain=0.0, tracking_error=None, smooth_scaling=False):
    """Integrate bimanual Cartesian DMP for one step with Euler integration.
    通过欧拉积分方法实现双臂机械臂笛卡尔DMP单步更新。

    Parameters
    ----------
    last_t : float
        Time at last step.  # 上一步的时间

    t : float
        Time at current step.  # 当前步时间

    current_y : array, shape (14,)
        Current position. Will be modified.  # 当前位置。将修改。

    current_yd : array, shape (12,)
        Current velocity. Will be modified.  # 当前速度。将被修改。

    goal_y : array, shape (14,)
        Goal position.  # 目标位置

    goal_yd : array, shape (12,)
        Goal velocity.  # 目标速度

    goal_ydd : array, shape (12,)
        Goal acceleration.  # 目标加速度。

    start_y : array, shape (14,)
        Start position.  # 起始位置

    start_yd : array, shape (12,)
        Start velocity.  # 起始速度

    start_ydd : array, shape (12,)
        Start acceleration.  # 起始加速度

    goal_t : float
        Time at the end.  # 结束时间

    start_t : float
        Time at the start.  # 起始时间

    alpha_y : float or array with shape (12,), optional (default: 25.0)
        Constant in transformation system.  # 变换系统中的常数

    beta_y : float or array with shape (12,), optional (default: 6.25)
        Constant in transformation system.  # 变换系统中的常数

    forcing_term : ForcingTerm
        Forcing term.  # 非线性力项

    coupling_term : CouplingTerm, optional (default: None)
        Coupling term. Must have a function coupling(y, yd) that returns
        additional velocity and acceleration.
        耦合项。必须有一个返回附加速度和加速度的函数 coupling(y,yd)。

    int_dt : float, optional (default: 0.001)
        Time delta used internally for integration.  # 内部积分时间差

    p_gain : float, optional (default: 0)
        Proportional gain for tracking error.  # 跟踪误差比例增益

    tracking_error : float, optional (default: 0)
        Tracking error from last step.  # 上一步的跟踪误差。

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.  # 平滑缩放选项
    """
    # 初始化：若当前时间未到起始时间，直接初始化当前位置和速度
    if t <= start_t:
        current_y[:] = start_y
        current_yd[:] = start_yd

    execution_time = goal_t - start_t  # 总执行时间

    current_ydd = np.empty_like(current_yd)  # 初始化当前加速度

    cd, cdd = np.zeros_like(current_yd), np.zeros_like(current_ydd)  # 初始化耦合速度和加速度

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd[:], cdd[:] = coupling_term.coupling(current_y, current_yd)

        z = forcing_term.phase(current_t)
        f = forcing_term.forcing_term(z).squeeze()
        if tracking_error is not None:
            cdd[pvs] += p_gain * tracking_error[pps] / dt
            for ops, ovs in ((slice(3, 7), slice(3, 6)),
                             (slice(10, 14), slice(9, 12))):
                cdd[ovs] += p_gain * pr.compact_axis_angle_from_quaternion(
                    tracking_error[ops]) / dt

        if smooth_scaling:
            smoothing = beta_y[pps] * (goal_y[pps] - start_y[pps]) * z
        else:
            smoothing = 0.0

        # position components
        current_ydd[pvs] = (
            alpha_y[pvs] * (
                beta_y[pvs] * (goal_y[pps] - current_y[pps])
                - execution_time * current_yd[pvs]
                - smoothing
            )
            + f[pvs]
            + cdd[pvs]
        ) / execution_time ** 2
        current_yd[pvs] += dt * current_ydd[pvs] + cd[pvs] / execution_time
        current_y[pps] += dt * current_yd[pvs]

        # orientation components
        for ops, ovs in ((slice(3, 7), slice(3, 6)),
                         (slice(10, 14), slice(9, 12))):
            if smooth_scaling:
                goal_y_minus_start_y = pr.compact_axis_angle_from_quaternion(
                    pr.concatenate_quaternions(goal_y[ops], pr.q_conj(start_y[ops])))
                smoothing = beta_y[ovs] * z * goal_y_minus_start_y
            else:
                smoothing = 0.0
            current_ydd[ovs] = (
                alpha_y[ovs] * (
                    beta_y[ovs] * pr.compact_axis_angle_from_quaternion(pr.concatenate_quaternions(
                        goal_y[ops], pr.q_conj(current_y[ops])))
                    - execution_time * current_yd[ovs]
                    - smoothing
                )
                + f[ovs]
                + cdd[ovs]
            ) / execution_time ** 2
            current_yd[ovs] += dt * current_ydd[ovs] + cd[ovs] / execution_time
            current_y[ops] = pr.concatenate_quaternions(
                pr.quaternion_from_compact_axis_angle(dt * current_yd[ovs]), current_y[ops])


DUAL_CARTESIAN_DMP_STEP_FUNCTIONS = {
    "python": dmp_step_dual_cartesian_python
}


try:
    from ..dmp_fast import dmp_step_dual_cartesian
    DUAL_CARTESIAN_DMP_STEP_FUNCTIONS["cython"] = dmp_step_dual_cartesian
    DEFAULT_DUAL_CARTESIAN_DMP_STEP_FUNCTION = "cython"
except ImportError:
    DEFAULT_DUAL_CARTESIAN_DMP_STEP_FUNCTION = "python"


class DualCartesianDMP(WeightParametersMixin, DMPBase):
    """Dual cartesian dynamical movement primitive.

    Each of the two Cartesian DMPs handles orientation and position separately.
    The orientation is represented by a quaternion.
    See :class:`CartesianDMP` for details about the equation of the
    transformation system.

    While the dimension of the state space is 14, the dimension of the
    velocity, acceleration, and forcing term is 12.
    两个笛卡尔 DMP 分别处理方向和位置。
    方向由四元数表示。
    有关变换系统方程的详细信息，请参见 :class:`CartesianDMP` 。

    状态空间的维度为 14，而速度、加速度和力项的维度为 12。

    Parameters
    ----------
    execution_time : float, optional (default: 1)
        Execution time of the DMP.

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.

    n_weights_per_dim : int, optional (default: 10)
        Number of weights of the function approximator per dimension.  # 每个维度的函数逼近器权重数。

    int_dt : float, optional (default: 0.001)
        Time difference for Euler integration.  # 欧拉积分的时差

    p_gain : float, optional (default: 0)
        Gain for proportional controller of DMP tracking error. # DMP 跟踪误差比例控制器的增益。
        The domain is [0, execution_time**2/dt].

    smooth_scaling : bool, optional (default: False)
        Avoids jumps during the beginning of DMP execution when the goal
        is changed and the trajectory is scaled by interpolating between
        the old and new scaling of the trajectory.
        通过对新旧轨迹缩放比例进行内插，避免在 DMP 执行之初目标发生变化和轨迹缩放时出现跳转。

    alpha_y : float or array-like, shape (12,), optional (default: 25.0)
        Parameter of the transformation system.

    beta_y : float or array-like, shape (12,), optional (default: 6.25)
        Parameter of the transformation system.

    Attributes  # 属性
    ----------
    execution_time_ : float
        Execution time of the DMP.

    dt_ : float
        Time difference between DMP steps. This value can be changed to adapt
        the frequency.
        DMP 步之间的时间差。该值可以更改，以调整频率。

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
    def __init__(self, execution_time=1.0, dt=0.01, n_weights_per_dim=10,
                 int_dt=0.001, p_gain=0.0, smooth_scaling=False, alpha_y=25.0, beta_y=6.25):
        super(DualCartesianDMP, self).__init__(14, 12)
        self._execution_time = execution_time
        self.dt_ = dt
        self.n_weights_per_dim = n_weights_per_dim
        self.int_dt = int_dt
        self.p_gain = p_gain
        self.smooth_scaling = smooth_scaling

        self._init_forcing_term()  # 初始化forcing_term()

        self.alpha_y = ensure_1d_array(alpha_y, 12, "alpha_y")  # 处理标量或类似数组的输入，确保其为 1D numpy 数组
        self.beta_y = ensure_1d_array(beta_y, 12, "beta_y")  # ndim:期望的数组维数；var_name:方面报错返回报错变量名

    def _init_forcing_term(self):
        alpha_z = canonical_system_alpha(0.01, self.execution_time_, 0.0)
        self.forcing_term = ForcingTerm(
            12, self.n_weights_per_dim, self.execution_time_, 0.0, 0.8,
            alpha_z)

    def get_execution_time_(self):
        return self._execution_time

    def set_execution_time_(self, execution_time):
        self._execution_time = execution_time
        weights = self.forcing_term.weights_
        self._init_forcing_term()
        self.forcing_term.weights_ = weights

    execution_time_ = property(get_execution_time_, set_execution_time_)

    def step(self, last_y, last_yd, coupling_term=None,
             step_function=DUAL_CARTESIAN_DMP_STEP_FUNCTIONS[
                 DEFAULT_DUAL_CARTESIAN_DMP_STEP_FUNCTION]):
        """DMP step.

        Parameters
        ----------
        last_y : array, shape (14,)
            Last state.

        last_yd : array, shape (12,)
            Last time derivative of state (velocity).

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.

        step_function : callable, optional (default: cython code if available)
            DMP integration function.

        Returns
        -------
        y : array, shape (14,)
            Next state.

        yd : array, shape (12,)
            Next time derivative of state (velocity).
        """
        assert len(last_y) == self.n_dims
        assert len(last_yd) == 12

        self.last_t = self.t
        self.t += self.dt_

        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        tracking_error = self.current_y - last_y
        for ops in (slice(3, 7), slice(10, 14)):
            tracking_error[ops] = pr.concatenate_quaternions(
                self.current_y[ops], pr.q_conj(last_y[ops]))
        self.current_y[:], self.current_yd[:] = last_y, last_yd
        step_function(
            self.last_t, self.t, self.current_y, self.current_yd,
            self.goal_y, self.goal_yd, self.goal_ydd,
            self.start_y, self.start_yd, self.start_ydd,
            self.execution_time_, 0.0,
            self.alpha_y, self.beta_y,
            self.forcing_term, coupling_term,
            self.int_dt,
            self.p_gain, tracking_error,
            self.smooth_scaling)

        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=None, coupling_term=None,
                  step_function=DEFAULT_DUAL_CARTESIAN_DMP_STEP_FUNCTION):
        """Run DMP open loop.

        Parameters
        ----------
        run_t : float, optional (default: execution_time)
            Run time of DMP. Can be shorter or longer than execution_time.

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.

        step_function : str, optional (default: 'cython' if available)
            DMP integration function. Possible options: 'python', 'cython'.

        Returns
        -------
        T : array, shape (n_steps,)
            Time for each step.

        Y : array, shape (n_steps, 14)
            State at each step.
        """
        try:
            step_function = DUAL_CARTESIAN_DMP_STEP_FUNCTIONS[step_function]
        except KeyError:
            raise ValueError(
                f"Step function must be in "
                f"{DUAL_CARTESIAN_DMP_STEP_FUNCTIONS.keys()}.")

        if run_t is None:
            run_t = self.execution_time_
        self.t = 0.0
        T = [self.t]
        Y = [np.copy(self.start_y)]
        y = np.copy(self.start_y)
        yd = np.copy(self.start_yd)
        while self.t < run_t:
            y, yd = self.step(y, yd, coupling_term, step_function)
            T.append(self.t)
            Y.append(np.copy(self.current_y))
        self.t = 0.0
        return np.array(T), np.vstack(Y)

    def imitate(self, T, Y, regularization_coefficient=0.0,
                allow_final_velocity=False):
        """Imitate demonstration.

        Target forces for the forcing term are computed for the positions
        in a similar way as in :func:`DMP.imitate`. For the orientations
        we adapt this to handle quaternions adequately.
        力项目标力的计算方法与 :func:`DMP.imitate` 中的方法类似。对于方向，
        我们将对其进行调整，以充分处理四元数。

        Parameters
        ----------
        T : array, shape (n_steps,)
            Time for each step.

        Y : array, shape (n_steps, 14)
            State at each step.

        regularization_coefficient : float, optional (default: 0)
            Regularization coefficient for regression.

        allow_final_velocity : bool, optional (default: False)
            Allow a final velocity.
        """
        self.forcing_term.weights_[:3, :] = dmp_imitate(
            T, Y[:, :3],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y[:3], beta_y=self.beta_y[:3],
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity,
            smooth_scaling=self.smooth_scaling)[0]
        self.forcing_term.weights_[3:6, :] = dmp_quaternion_imitation(
            T, Y[:, 3:7],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y[3:6], beta_y=self.beta_y[3:6],
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity,
            smooth_scaling=self.smooth_scaling)[0]
        self.forcing_term.weights_[6:9, :] = dmp_imitate(
            T, Y[:, 7:10],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y[6:9], beta_y=self.beta_y[6:9],
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity,
            smooth_scaling=self.smooth_scaling)[0]
        self.forcing_term.weights_[9:12, :] = dmp_quaternion_imitation(
            T, Y[:, 10:14],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y[9:12], beta_y=self.beta_y[9:12],
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity,
            smooth_scaling=self.smooth_scaling)[0]

        self.configure(start_y=Y[0], goal_y=Y[-1])
