"""
===================
Two Coupled 1D DMPs
双耦合1D DMPs
===================

Two 1D DMPs are spatially coupled with a virtual spring that forces them to
keep a distance. One of them is the leader DMP and the other one is the
follower DMP.
两个一维 DMP 在空间上通过虚拟弹簧耦合，迫使它们保持距离。
保持一定距离。其中一个是领跑者 DMP，另一个是
跟随者 DMP。
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP, CouplingTermPos1DToPos1D


# 参数设置
dt = 0.01
execution_time = 2.0
desired_distance = 0.5
dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt, n_weights_per_dim=200)
coupling_term = CouplingTermPos1DToPos1D(
    desired_distance=desired_distance, lf=(1.0, 0.0), k=1.0)  # k:耦合位置的虚拟弹簧常数

T = np.linspace(0.0, execution_time, 101)  # linspace包含stop值、arange:不包含
Y = np.empty((len(T), 2))
Y[:, 0] = np.cos(2.5 * np.pi * T)  # 2.5个循环周期
Y[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)  # 1.5个循环周期
dmp.imitate(T, Y)


# 绘图
fig = plt.figure(figsize=(10, 5)) # 创建图窗

# 设置子图属性
ax1 = fig.add_subplot(131)
ax1.set_title("Dimension 1")
ax1.set_ylim((-3, 3))
ax2 = fig.add_subplot(132)
ax2.set_title("Dimension 2")
ax2.set_ylim((-3, 3))
ax1.plot(T, Y[:, 0], label="Demo1")  # 子图1，标记曲线的起始点和结束点
ax1.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
ax2.plot(T, Y[:, 1], label="Demo2")
ax2.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])  # 子图2，标记曲线的起始点和结束点

# Reproduction曲线
dmp.configure(start_y=Y[0], goal_y=Y[-1])  # 设置元参数
T, Y = dmp.open_loop()  # 开环运行DMP
ax1.plot(T, Y[:, 0], label="Reproduction1")
ax1.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
ax2.plot(T, Y[:, 1], label="Reproduction2")
ax2.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

# Coupled曲线
dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop(coupling_term=coupling_term)
ax1.plot(T, Y[:, 0], label="Coupled 1")
ax2.plot(T, Y[:, 1], label="Coupled 2")
ax1.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
ax2.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

ax1.legend(loc="best")
ax2.legend(loc="best")

# 子图3
ax = fig.add_subplot(133)
ax.set_title("Distance")
ax.set_ylim((-3, 3))
ax.plot(T, Y[:, 1] - Y[:, 0], label="Actual distance")
ax.plot([T[0], T[-1]], [desired_distance, desired_distance],
        label="Desired distance")

ax.legend(loc="best")  # best:位置自适应

plt.show()
