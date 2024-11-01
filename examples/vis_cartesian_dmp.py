"""
====================
Cartesian DMP on UR5
====================

A trajectory is created manually, imitated with a Cartesian DMP, converted
to a joint trajectory by inverse kinematics, and executed with a UR5.
手动创建轨迹，用笛卡尔 DMP 进行模仿，通过逆运动学转换为关节轨迹，再用 UR5 执行。
通过逆运动学转换为关节轨迹，然后用 UR5 执行。
"""
print(__doc__)

import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
from movement_primitives.dmp import CartesianDMP


def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph


rotation_angle = np.deg2rad(45)
n_steps = 201

# src ='../examples/data/urdf/ur5.urdf'
with open("examples/data/urdf/ur5.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="examples/data/urdf/")  #  Kinematics:机器人运动学 mesh_path:用于搜索 URDF 中定义的网格的路径。如果设置为 "无"，网格将被忽略
chain = kin.create_chain(
    ["ur5_shoulder_pan_joint", "ur5_shoulder_lift_joint", "ur5_elbow_joint",
     "ur5_wrist_1_joint", "ur5_wrist_2_joint", "ur5_wrist_3_joint"],
    "ur5_base_link", "ur5_tool0")

q0 = np.array([0.0, -0.5, 0.8, -0.5, 0, 0])
# q0 = np.array([0.0, 1.2, 0, 0, 0, 0])
chain.forward(q0)  # 正运动学，从末端执行器到基准框架的转换

ee2base_start = kin.tm.get_transform("ur5_tool0", "ur5_base_link")  # tm: FastUrdfTransformManager 转换管理器

rotation_axis = -pr.unity
start2end = pt.rotate_transform(np.eye(4), pr.matrix_from_compact_axis_angle(
    rotation_axis * rotation_angle))
ee2base_end = pt.concat(ee2base_start, start2end)  # concat:变换矩阵相乘

start = pt.exponential_coordinates_from_transform(ee2base_start)  # 根据变换矩阵计算指数坐标
end = pt.exponential_coordinates_from_transform(ee2base_end)

T = np.linspace(0, 1, n_steps)
# print(start[np.newaxis].shape, end[np.newaxis].shape, T[:, np.newaxis].shape, T[:, np.newaxis] * (end[np.newaxis] - start[np.newaxis]))
trajectory = start[np.newaxis] + T[:, np.newaxis] * (end[np.newaxis] - start[np.newaxis])  # np.newaxis:增加新的维度  trajectory:shape(n_steps=201, start.shape=6)

dt = 0.01
execution_time = (n_steps - 1) * dt
T = np.linspace(0, execution_time, n_steps)  # 时间轴
Y = ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(trajectory))  # 从齐次矩阵中获取位置序列和四元数。Returns:P:array,shape(n_steps, 7),顺序为(x、y、z、qw、qx、qy、qz)
dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)  # 笛卡尔动态运动基元
dmp.imitate(T, Y)
_, Y = dmp.open_loop()  # Y:array, shape (n_steps, 7) 每一步的状态

trajectory = ptr.transforms_from_pqs(Y)  # 从位置和四元数中获取齐次矩阵序列 Returns:A2Bs:array,shape(…, 4, 4)

random_state = np.random.RandomState(0)
joint_trajectory = chain.inverse_trajectory(
    trajectory, q0, random_state=random_state)  # 计算轨迹的逆运动学

fig = pv.figure()
fig.plot_transform(s=0.3)  # 绘制坐标系 s:将绘制的轴线和角度的缩放比例

graph = fig.plot_graph(
    kin.tm, "ur5_base_link", show_visuals=False, show_collision_objects=True,
    show_frames=True, s=0.1, whitelist=["ur5_base_link", "ur5_tool0"])  # 绘制连接框架图 tm:转换管理器 whitelist:应显示的坐标系列表

fig.plot_transform(ee2base_start, s=0.15)
fig.plot_transform(ee2base_end, s=0.15)
fig.plot_transform(trajectory[-1], s=0.15)
fig.plot_transform(trajectory[-1], s=0.15)

pv.Trajectory(trajectory, s=0.05).add_artist(fig)

fig.view_init()  # 设置轴的仰角和方位角，默认值分别为-60，30
fig.animate(
    animation_callback, len(trajectory), loop=True,
    fargs=(graph, chain, joint_trajectory))  # animate(callback, n_frames, loop=False, fargs=())
# callback:可调用循环调用的回调函数，用于更新几何图形;n_frames:总帧数;loop:用来控制动画是否循环;fargs:将被传递给回调函数的参数
fig.show()
