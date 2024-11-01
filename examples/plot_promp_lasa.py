"""
============================
LASA Handwriting with ProMPs
============================

The LASA Handwriting dataset learned with ProMPs. The dataset consists of
2D handwriting motions. The first and third column of the plot represent
demonstrations and the second and fourth column show the imitated ProMPs
with 1-sigma interval.
使用 ProMPs 学习的 LASA 手写数据集。该数据集包括
二维手写动作。图中第一列和第三列表示
示例，第二列和第四列表示模仿的 ProMPs
以 1-sigma 为间隔。
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.data import load_lasa
from movement_primitives.promp import ProMP


def draw(T, X, idx, axes, shape_name):
    """根据给定数据绘制轨迹。

    Parameters
    ----------
    T : array, shape (n_demos, n_steps)
        Times

    X : array, shape (n_demos, n_steps, n_dims) n_demos:曲线数量(7); n_steps:时间步(1000)
        Positions

    idx : int
        第idx个子图

    axes : `~matplotlib.axes.Axes` or array of Axes
        轴对象

    shape_name : string
        Name of the Matlab file from which we load the demonstrations 演示
        (without suffix). suffix:后缀

    Returns
    -------
    None
    """
    h = int(idx / width)
    w = int(idx % width) * 2
    axes[h, w].set_title(shape_name)
    axes[h, w].plot(X[:, :, 0].T, X[:, :, 1].T)
    # axes[h, w].plot(X[:, :, 0], X[:, :, 1])

    promp = ProMP(n_weights_per_dim=30, n_dims=X.shape[2]) # ProMP(n_dims[, n_weights_per_dim]) n_dims:状态空间维度；n_weights_per_dim:每个维度的函数逼近器权重数
    promp.imitate(T, X)
    # print('T[0].shape:', T[0].shape)
    mean = promp.mean_trajectory(T[0])  # 获取 ProMP 的平均轨迹 Returns: Y: array, shape (n_steps, n_dims)
    std = np.sqrt(promp.var_trajectory(T[0])) # 获取 ProMP 的轨迹方差 Returns: var: array, shape (n_steps, n_dims)

    # 绘制平均轨迹及方差范围
    # print(type(mean), mean.shape, mean)
    # print(type(std), std.shape, std)
    axes[h, w + 1].plot(mean[:, 0], mean[:, 1], c="r")
    axes[h, w + 1].plot(mean[:, 0] - std[:, 0], mean[:, 1] - std[:, 1], c="g")
    axes[h, w + 1].plot(mean[:, 0] + std[:, 0], mean[:, 1] + std[:, 1], c="g")

    # 设置子图坐标轴范围
    axes[h, w + 1].set_xlim(axes[h, w].get_xlim())
    axes[h, w + 1].set_ylim(axes[h, w].get_ylim())
    axes[h, w].get_yaxis().set_visible(False)
    axes[h, w].get_xaxis().set_visible(False)
    axes[h, w + 1].get_yaxis().set_visible(False)
    axes[h, w + 1].get_xaxis().set_visible(False)


# 绘图长宽的手写曲线数量
width = 2
height = 5

# 绘图
fig, axes = plt.subplots(int(height), int(width * 2))  # ax:Axes or array of Axes

for i in range(width * height):
    T, X, Xd, Xdd, dt, shape_name = load_lasa(i)
    draw(T, X, i, axes, shape_name)
plt.tight_layout()
plt.show()
