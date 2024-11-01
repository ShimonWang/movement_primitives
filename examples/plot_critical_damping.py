"""
================
Critical Damping
临界阻尼
================

The transformation system of a DMP converges to the goal and the convergence is
modeled as a spring-damper system. For an optimal convergence, the constants
defining the spring-damper system (spring constant k and damping coefficient c)
have to be set to critical damping for optimal convergence, that is, as quickly
as possible without overshooting. To illustrate this, we use a standalone
spring-damper system and explore several values for these parameters.
DMP 的变换系统会向目标收敛，收敛过程被模拟为弹簧-阻尼系统。
建模为弹簧-阻尼系统。为了达到最佳收敛效果，定义弹簧-阻尼系统的常数
（弹簧常数 k 和阻尼系数 c）
必须设置为最佳收敛的临界阻尼，即在不超调的情况下尽可能快地
而不会出现过冲。为了说明这一点，我们使用一个独立的
弹簧阻尼系统，并探索这些参数的几种值。
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.spring_damper import SpringDamper


k = 100  # 弹簧常数
start_y = np.zeros(1)  # 起始点
goal_y = np.ones(1)  # 终止点
for c in [10, 20, 40]:
    attractor = SpringDamper(n_dims=1, k=k, c=c, dt=0.01)  # SpringDamper:类似于没有力项的DMP;c:阻尼系数
    attractor.configure(start_y=start_y, goal_y=goal_y)
    T, Y = attractor.open_loop(run_t=2.0)  # Y:shape(201,1)
    plt.plot(T, Y[:, 0], label=f"$k={k}, c={c}$")
plt.scatter(1.0, 1.0, marker="*", s=200, label="Goal")
plt.legend(loc="best")
plt.title(r"Condition for critical damping: $c = 2 \sqrt{k}$")
plt.show()
