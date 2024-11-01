"""
=======================
Minimum Jerk Trajectory
最小跃变轨迹
=======================

An example for a minimum jerk trajectory is displayed in the following plot.
下图显示了 minimum jerk trajectory 最小跃变轨迹 的示例。
"""
print(__doc__)

import matplotlib.pyplot as plt
from movement_primitives.data import generate_minimum_jerk

# minimum jerk trajectory
X, Xd, Xdd = generate_minimum_jerk([0], [1])

# 绘图
plt.figure()
plt.subplot(311)
plt.ylabel("$x$")
plt.plot(X[:, 0]) # plt.plot(X[:])
plt.subplot(312)
plt.ylabel("$\dot{x}$")
plt.plot(Xd[:, 0])
plt.subplot(313)
plt.xlabel("$t$")
plt.ylabel("$\ddot{x}$")
plt.plot(Xdd[:, 0])
plt.show()
