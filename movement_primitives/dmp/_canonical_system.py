import numpy as np


def canonical_system_alpha(goal_z, goal_t, start_t):
    r"""Compute parameter alpha of canonical system.  # 计算正则系统的参数 alpha。

    The parameter alpha is computed such that a specific phase value goal_z
    is reached at goal_t. The canonical system is defined according to [1]_,
    even though we compute a different value for alpha.
    参数 alpha 的计算方法是，在goal_t 处达到特定的相位值 goal_z。
    尽管我们计算的 alpha 值不同，但正则系统是根据 [1]_ 来定义的。

    Parameters
    ----------
    goal_z : float
        Value of phase variable at the end of the execution (> 0).  # 执行结束时相位变量的值（> 0）。

    goal_t : float
        Time at which the execution should be done. Make sure that
        goal_t > start_t.

    start_t : float
        Time at which the execution should start.

    Returns
    -------
    alpha : float
        Value of the alpha parameter of the canonical system.

    Raises
    ------
    ValueError
        If input values are invalid.

    References
    ----------
    .. [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., Schaal, S.
       (2013). Dynamical Movement Primitives: Learning Attractor Models for
       Motor Behaviors. Neural Computation 25 (2), 328-373. DOI:
       10.1162/NECO_a_00393,
       https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
    """
    if goal_z <= 0.0:
        raise ValueError("Final phase must be > 0!")
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    return float(-np.log(goal_z))


def phase(t, alpha, goal_t, start_t):
    r"""Map time to phase.

    According to [1]_, the differential Equation

    .. math::

        \tau \dot{z} = -\alpha_z z

    describes the evolution of the phase variable z. Starting from the initial
    position :math:`z_0 = 1`, the phase value converges monotonically to 0.
    Instead of using an iterative procedure to calculate the current value of
    z, it is computed directly through
    从初始位置 :math:`z_0 = 1` 开始，相位值单调地趋近于 0。
    计算 z 的当前值时，无需使用迭代程序，而是直接通过

    .. math::

        z(t) = \exp( - \frac{\alpha_z}{\tau} t)

    Parameters
    ----------
    t : float
        Current time.

    alpha : float
        Value of the alpha parameter of the canonical system.

    goal_t : float
        Time at which the execution should be done.

    start_t : float
        Time at which the execution should start.

    Returns
    -------
    z : float
        Value of phase variable.

    References
    ----------
    .. [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., Schaal, S.
       (2013). Dynamical Movement Primitives: Learning Attractor Models for
       Motor Behaviors. Neural Computation 25 (2), 328-373. DOI:
       10.1162/NECO_a_00393,
       https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
    """
    execution_time = goal_t - start_t
    return np.exp(-alpha * (t - start_t) / execution_time)  # execution_time \tau 时间放缩因子
