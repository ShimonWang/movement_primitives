"""
========================
Obstacle Avoidance in 3D
三维避障
========================

Plots a DMP in 3D that goes through a point obstacle without obstacle
avoidance. Then we start use the same DMP to start from multiple random
start positions with an activated coupling term for obstacle avoidance.
在三维空间中绘制一个通过点障碍物而不避障的 DMP。
然后，我们开始使用同一 DMP 从多个随机起始位置出发，并激活耦合项以避开障碍物。
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.plot_utils as ppu
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance3D


# 设置参数
execution_time = 1.0
start_y = np.zeros(3)
goal_y = np.ones(3)
obstacle = 0.5 * np.ones(3)
random_state = np.random.RandomState(42)

# 创建DMP对象
dmp = DMP(n_dims=len(start_y), execution_time=execution_time,
          n_weights_per_dim=10)
dmp.configure(start_y=start_y, goal_y=goal_y)

# 绘图
ax = ppu.make_3d_axis(1)
T, Y = dmp.open_loop()
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], c="g")
ax.scatter(start_y[0], start_y[1], start_y[2], c="r")
ax.scatter(goal_y[0], goal_y[1], goal_y[2], c="g")
ax.scatter(obstacle[0], obstacle[1], obstacle[2], c="y")
for _ in range(20):
    start_y_random = 0.2 + 0.2 * random_state.randn(3)
    coupling_term = CouplingTermObstacleAvoidance3D(obstacle)
    dmp.configure(start_y=start_y_random)
    T, Y = dmp.open_loop(coupling_term=coupling_term)
    ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], c="b")
    ax.scatter(start_y_random[0], start_y_random[1], start_y_random[2], c="b")
ax.set_xlim((-0.2, 1.2))
ax.set_ylim((-0.2, 1.2))
ax.set_zlim((-0.2, 1.2))
plt.show()
