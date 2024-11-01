"""
==================
Effect of DMP Gain
==================

Demonstrates how modifying DMP gains (alpha_y, beta_y) affects 
the resulting trajectory reproduction.
演示修改 DMP 增益（alpha_y、beta_y）如何影响轨迹再现。
由此产生的轨迹再现。
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP


# 参数设置
dt = 0.01

dmp1 = DMP(
    n_dims=2,
    execution_time=1.0,
    dt=dt,
    n_weights_per_dim=10,
    int_dt=0.0001,
    alpha_y=np.array([25.0, 25.0]),
    beta_y=np.array([6.25, 6.25]),
)

dmp2 = DMP(
    n_dims=2,
    execution_time=1.0,
    dt=dt,
    n_weights_per_dim=10,
    int_dt=0.0001,
    alpha_y=np.array([25.0, 10.0]),  # note different alpha_y
    beta_y=np.array([6.25, 3.0]),  # note different beta_y
)

# 示教轨迹数据创建
T = np.linspace(0.0, 1.0, 101)
Y = np.empty((101, 2))
Y[:, 0] = np.cos(np.pi * T)  # x=cos(pi*t)
Y[:, 1] = np.sin(np.pi * T)  # y=sin(pi*t)

# 绘图
plt.plot(Y[:, 0], Y[:, 1], label="Demo")
dmps = [dmp1, dmp2]
for i, dmp in enumerate(dmps):

    dmp.imitate(T, Y)
    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    _, Y_ = dmp.open_loop()
    plt.plot(Y_[:, 0], Y_[:, 1], label=f"Reproduction {i+1}")

# 网格等图窗属性设置
plt.grid()
plt.gca().set_aspect("equal", "box")  #  获取当前坐标轴;'equal':即 x 和 y 的缩放比例相同
plt.legend()
plt.xlabel("x")
plt.ylabel("y")

plt.show()
