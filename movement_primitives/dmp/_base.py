import numpy as np
from ..base import PointToPointMovement


class DMPBase(PointToPointMovement):
    """Base class of Dynamical Movement Primitives (DMPs).  # 动态运动基元的基类

    Parameters
    ----------
    n_pos_dims : int
        Number of dimensions of the position that will be controlled.  # 将要控制的位置的维度数

    n_vel_dims : int
        Number of dimensions of the velocity that will be controlled.  # 将要控制的速度的维度数
    """
    def __init__(self, n_pos_dims, n_vel_dims):
        super(DMPBase, self).__init__(n_pos_dims, n_vel_dims)  # Python3中可以省略super中参数 DMPBase, self

        self.initialized = False

    def reset(self):
        """Reset DMP to initial state and time.  # 将DMP重置为初始状态和时间。"""
        self.t = 0.0
        self.last_t = None
        self.current_y = np.copy(self.start_y)
        self.current_yd = np.copy(self.start_yd)


class WeightParametersMixin:
    """Mixin class providing common access methods to forcing term weights.

    This can be used, for instance, for black-box optimization of the weights
    with respect to some cost / objective function in a reinforcement learning
    setting.
    Mixin 类提供了力项权重的常用访问方法。

    例如，这可用于在强化学习设置中，根据某些成本/目标函数对权重进行黑盒优化。
    """
    def get_weights(self):
        """Get weight vector of DMP.

        Returns
        -------
        weights : array, shape (N * n_weights_per_dim,)
            Current weights of the DMP. N depends on the type of DMP
        """
        return self.forcing_term.weights_.ravel()

    def set_weights(self, weights):
        """Set weight vector of DMP.

        Parameters
        ----------
        weights : array, shape (N * n_weights_per_dim,)
            New weights of the DMP. N depends on the type of DMP
        """
        self.forcing_term.weights_[:, :] = weights.reshape(*self.forcing_term.shape)

    @property
    def n_weights(self):
        """Total number of weights configuring the forcing term.
        配置力项的总权重数
        """
        return np.prod(self.forcing_term.shape)  # np.prod:返回给定轴上的数组元素的乘积
