"""
===================================
Pose Coupling of Dual Cartesian DMP
===================================

A dual Cartesian DMP is learned from an artificially generated demonstration
and replayed with and without a coupling of the pose of the two end effectors.

The red line indicates the DMP without coupling term and the orange line marks
the DMP with coupling term.
从人工生成的演示中学习双笛卡尔 DMP
并在两个末端效应器的姿势耦合和不耦合的情况下进行重放。

红线表示无耦合项的 DMP，橙线表示有耦合项的 DMP。
"""
print(__doc__)


import warnings
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DualCartesianDMP, CouplingTermDualCartesianPose
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from pytransform3d.urdf import UrdfTransformManager
from movement_primitives.testing.simulation import SimulationMockup


# 参数设置
dt = 0.01
int_dt = 0.001
execution_time = 1.0

desired_distance = np.array([  # right arm to left arm
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.2],
    [0.0, 0.0, 0.0, 1.0]
])
desired_distance[:3, :3] = pr.matrix_from_compact_axis_angle([np.deg2rad(180), 0, 0])  # 根据紧凑轴角计算旋转矩阵
ct = CouplingTermDualCartesianPose(  # 双笛卡尔 DMP 的耦合相对姿势 if:二进制值，表示将对哪些 DMP 进行调整。变量 lf 定义了领导者与跟随者的关系。
    desired_distance=desired_distance, couple_position=True, couple_orientation=True,
    lf=(1.0, 0.0), k=1, c1=0.1, c2=1000)  # c2=10000 in simulation

rh5 = SimulationMockup(dt=dt)  # 开环运行步进程序

Y = np.zeros((1001, 14))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)  # 双曲正切函数
radius = 0.5

circle1 = radius * np.cos(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
circle2 = radius * np.sin(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
# 替换Y0,1,2三列
Y[:, 0] = circle1
Y[:, 1] = 0.55
Y[:, 2] = circle2
R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])  # 根据轴角计算旋转矩阵
R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(0)])
# introduces coupling error (default goal: -90; error at: -110)
R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-110)])
q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))  # 根据旋转矩阵计算四元数
q_end = -pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
for i, t in enumerate(T):
    Y[i, 3:7] = pr.quaternion_slerp(q_start, q_end, t)

circle1 = radius * np.cos(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
circle2 = radius * np.sin(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
Y[:, 7] = circle1
Y[:, 8] = 0.55
Y[:, 9] = circle2
R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-180)])
R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-270)])
q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
q_end = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
for i, t in enumerate(T):
    Y[i, 10:] = pr.quaternion_slerp(q_start, q_end, t)

tm = UrdfTransformManager()
try:
    with open("abstract-urdf-gripper/urdf/GripperLeft.urdf", "r") as f:
        tm.load_urdf(f.read(), mesh_path="abstract-urdf-gripper/urdf/")
    with open("abstract-urdf-gripper/urdf/GripperRight.urdf", "r") as f:
        tm.load_urdf(f.read(), mesh_path="abstract-urdf-gripper/urdf/")
except FileNotFoundError:
    warnings.warn("URDF not found")

tm.add_transform("ALWristPitch_Link", "base", np.eye(4))
tm.add_transform("ARWristPitch_Link", "base", np.eye(4))

dmp = DualCartesianDMP(
    execution_time=execution_time, dt=dt,
    n_weights_per_dim=10, int_dt=int_dt, p_gain=0.0)
dmp.imitate(T, Y)
ax = pr.plot_basis(ax_s=0.8, s=0.1)
for coupling_term, color in [(ct, "orange"), (None, "red")]:
    dmp.reset()

    rh5.goto_ee_state(Y[0])
    desired_positions, positions, desired_velocities, velocities = \
        rh5.step_through_cartesian(dmp, Y[0], np.zeros(12), execution_time,
                                   coupling_term=coupling_term)
    P = np.asarray(positions)
    V = np.asarray(velocities)

    ptr.plot_trajectory(ax=ax, P=P[:, :7], s=0.05, color=color, show_direction=False)
    ptr.plot_trajectory(ax=ax, P=P[:, 7:], s=0.05, color=color, show_direction=False)
    for t in range(0, len(P), 50):
        gripper_left2base = pt.transform_from_pq(P[t, :7])
        gripper_right2base = pt.transform_from_pq(P[t, 7:])
        tm.add_transform("ALWristPitch_Link", "base", gripper_left2base)
        tm.add_transform("ARWristPitch_Link", "base", gripper_right2base)
        ax = tm.plot_visuals(frame="base", ax=ax)
    ax.view_init(elev=0, azim=90)
plt.show()
