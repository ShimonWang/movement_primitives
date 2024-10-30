"""
======================
DMP as Potential Field
======================

A Dynamical Movement Primitive defines a potential field that superimposes
several components: transformation system (goal-directed movement), forcing
term (learned shape), and coupling terms (e.g., obstacle avoidance).
"""
print(__doc__)  # __doc__用于获取函数或模块的注释内容


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance2D
from movement_primitives.dmp_potential_field import plot_potential_field_2d

# 依次创建创建初始状态、目标状态、ndarray数组对象
start_y = np.array([0, 0], dtype=float)  # 初始状态
goal_y = np.array([1, 1], dtype=float)  # 目标状态
obstacle = np.array([0.85, 0.5])  # 障碍物状态
random_state = np.random.RandomState(1)  # 随机状态  伪随机数生成器

# 创建一个 2 维的 DMP 对象dmp，维度数为 2，每个维度有 10 个权重，时间步长 dt=0.01，执行时间 execution_time=1.0
dmp = DMP(n_dims=2, n_weights_per_dim=10, dt=0.01, execution_time=1.0)# n_dims:int 状态空间维度;
# n_weights_per_dim:int,默认值10 每个维度的函数逼近器权重数；dt:float 默认值0.01; execution_time:float 默认值1 DMP的执行时间;
dmp.configure(start_y=start_y, goal_y=goal_y)# start_yarray, shape(n_dims,); goal_yarray, shape (n_dims,)

# 创建一个 2 维的 DMP 对象dmp_ft，维度数为 2，每个维度有 10 个权重，时间步长 dt=0.01，执行时间 execution_time=1.0
dmp_ft = DMP(n_dims=2, n_weights_per_dim=10, dt=0.01, execution_time=1.0)
dmp_ft.forcing_term.weights_[:, :] = random_state.randn(
    *dmp_ft.forcing_term.weights_.shape) * 500.0  # weights_:shape(n_dims, n_weights_per_dim)
dmp_ft.configure(start_y=start_y, goal_y=goal_y)

# 创建一个 2 维的 DMP 对象dmp_ct，维度数为 2，每个维度有 10 个权重，时间步长 dt=0.01，执行时间 execution_time=1.0
dmp_ct = DMP(n_dims=2, n_weights_per_dim=10, dt=0.01, execution_time=1.0)
dmp_ct.forcing_term.weights_[:, :] = dmp_ft.forcing_term.weights_[:, :]
dmp_ct.configure(start_y=start_y, goal_y=goal_y)
coupling_term = CouplingTermObstacleAvoidance2D(obstacle)  # class movement_primitives.dmp.CouplingTermObstacleAvoidance2D(obstacle_position, gamma=1000.0, beta=6.366197723675814, fast=False)

# 绘图参数
n_rows, n_cols = 2, 4
n_subplots = n_rows * n_cols
x_range = -0.2, 1.2  # x_range:{tuple:2}
y_range = -0.2, 1.2

position = np.copy(start_y)  # [0., 0.]
velocity = np.zeros_like(start_y)  # numpy.zeros_like 用于创建一个与给定数组具有相同形状的数组，数组元素以 0 来填充。

position_ft = np.copy(start_y)
velocity_ft = np.zeros_like(start_y)

position_ct = np.copy(start_y)
velocity_ct = np.zeros_like(start_y)

plt.figure(figsize=(12, 6))
positions = [position]  # positions={list:1}[[0. 0.]]
positions_ft = [position_ft]
positions_ct = [position_ct]
for i in range(n_subplots):
    ax = plt.subplot(n_rows, n_cols, i + 1, aspect="equal")
    ax.set_title(f"t = {dmp.t:.02f}", backgroundcolor="#ffffffff", y=0.05)  # #ffffffff 不透明白色

    # 绘制二维势场
    plot_potential_field_2d(
        ax, dmp_ct, x_range=x_range, y_range=y_range, n_ticks=15,
        obstacle=obstacle)
    plt.plot(start_y[0], start_y[1], "o", color="b", markersize=10)
    plt.plot(goal_y[0], goal_y[1], "o", color="g", markersize=10)
    plt.plot(obstacle[0], obstacle[1], "o", color="y", markersize=10)

    path = np.array(positions)  # path={ndarray:{1,2}}
    plt.plot(path[:, 0], path[:, 1], lw=5, color="g", label="Transformation System")
    path_ft = np.array(positions_ft)
    plt.plot(path_ft[:, 0], path_ft[:, 1], lw=5, color="r", label="+ Forcing Term")
    path_ct = np.array(positions_ct)
    plt.plot(path_ct[:, 0], path_ct[:, 1], lw=5, color="y", label="+ Obstacle Avoidance")

    # 设置图坐标轴范围
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    plt.setp(ax, xticks=(), yticks=())
    if i == 0:
        ax.legend(loc="upper left")

    if i == n_subplots - 1:
        break

    # 绘制Transform System, Forcing Term, Obstacle Term 的前(1 + i) / (n_subplots - 1)秒轨迹
    while dmp.t <= dmp.execution_time_ * (1 + i) / (n_subplots - 1):
        position, velocity = dmp.step(position, velocity)  # abstract y, yd = step(last_y, last_yd)
        positions.append(position)
        position_ft, velocity_ft = dmp_ft.step(position_ft, velocity_ft)
        positions_ft.append(position_ft)
        position_ct, velocity_ct = dmp_ct.step(
            position_ct, velocity_ct, coupling_term=coupling_term)
        positions_ct.append(position_ct)

# 调整图窗并展示
plt.subplots_adjust(
    left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)  # wspace/hspace:子图块的填充 宽度/高度 ，占坐标轴平均 宽度/高度 的百分比
plt.show()
