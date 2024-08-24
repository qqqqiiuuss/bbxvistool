
import numpy as np
def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

# 示例用法
T_lidar_camera = [
    1.357013632401,
    4.425804820052395,
    3.8965668389678143,
    -0.380650928236051,
    0.6868913889320076,
    -0.44467397488791593,
    0.4307553211528225
]

t = T_lidar_camera[:3]  # 平移向量
q = T_lidar_camera[3:]  # 四元数

R = quaternion_to_rotation_matrix(q)  # 旋转矩阵
print("平移向量 t:", t)
print("旋转矩阵 R:\n", R)
