import torch
import numpy as np


def foot_position_in_hip_frame(angles, l_hip_sign=1):
    """ Left legs: l_hip_sign=1; right legs: l_hip_sign=-1 """
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float)
    l_hip = 0.08505 * l_hip_sign  # TODO
    l_thigh = 0.213  # 0.213
    l_calf = 0.2

    theta_ab, theta_hip, theta_knee = angles[:, 0].unsqueeze(-1), angles[:, 1].unsqueeze(-1), angles[:, 2].unsqueeze(-1)

    leg_distance = torch.sqrt(l_thigh ** 2 + l_calf ** 2 + 2 * l_thigh * l_calf * torch.cos(theta_knee))
    eff_swing = theta_hip + theta_knee / 2

    off_x_hip = -leg_distance * torch.sin(eff_swing)
    off_z_hip = -leg_distance * torch.cos(eff_swing)
    off_y_hip = l_hip

    off_x = off_x_hip
    off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
    off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip

    foot_pos = torch.cat((off_x, off_y, off_z), dim=1)
    return foot_pos


def angles_in_hip_frame_to_feet_pos(joint_angles):
    """
    Description: Calculate foot pos of four legs based on the given joint angles
    -----------
    Input: joint angles of four legs (FR/FL/RR/RL) -> torch.Tensor Shape([num_envs, 12])
    Output: foot position of four legs (FR/FL/RR/RL) -> torch.Tensor Shape([num_envs, 12])
    """
    if not isinstance(joint_angles, torch.Tensor):
        joint_angles = torch.tensor(joint_angles, device='cuda', dtype=torch.float)
    num_envs = joint_angles.shape[0]
    feet_pos = torch.zeros(num_envs, 12, device='cuda', dtype=torch.float)
    for i in range(4):
        feet_pos[:, 3 * i:3 * i + 3] = foot_position_in_hip_frame(joint_angles[:, 3 * i:3 * i + 3],
                                                                               l_hip_sign=(-1) ** (i + 1))
    return feet_pos  # torch.Size([num_envs, 12])


def foot_pos_in_hip_frame_to_joint_angle(foot_position, l_hip_sign=1):
    """
    foot_position in the hip frame (Note: not in base frame!) -> torch.Size([num_envs, 3])
    """
    # default "length" of each link
    l_hip = 0.08505 * l_hip_sign  # TODO
    l_thigh = 0.213  # 0.213
    l_calf = 0.2

    # torch.Size([num_envs, 1])
    x, y, z = foot_position[:, 0].unsqueeze(-1), foot_position[:, 1].unsqueeze(-1), foot_position[:, 2].unsqueeze(
        -1)  # torch.Size([num_envs, 1])

    # angle of knee
    theta_knee = - torch.acos(
        (torch.sum(torch.square(foot_position), dim=1, keepdim=True) - l_hip ** 2 - l_calf ** 2 - l_thigh ** 2) /
        (2 * l_calf * l_thigh))  # torch.Size([num_envs, 1])

    l = torch.sqrt(
        l_thigh ** 2 + l_calf ** 2 + 2 * l_thigh * l_calf * torch.cos(theta_knee))  # torch.Size([num_envs, 1])

    # angle of hip joint
    theta_hip = torch.asin(- torch.div(x, l)) - theta_knee / 2

    c1 = l_hip * y - torch.mul(torch.mul(l, torch.cos(theta_hip + theta_knee / 2)), z)  # torch.Size([num_envs, 1])
    s1 = torch.mul(torch.mul(l, torch.cos(theta_hip + theta_knee / 2)), y) + l_hip * z  # torch.Size([num_envs, 1])

    theta_ab = torch.atan2(s1, c1)  # torch.Size([num_envs, 1])

    joint_angles = torch.cat((theta_ab, theta_hip, theta_knee), dim=1)
    return joint_angles


def feet_pos_in_hip_frame_to_angles(feet_pos):
    """
    Description: Calculate joint angles based on the given foot position of four legs
    -----------
    Input: foot position of four legs (FR/FL/RR/RL) -> torch.Tensor Shape([num_envs, 12])
    Output: joint angles of four legs (FR/FL/RR/RL) -> torch.Tensor Shape([num_envs, 12])
    """
    num_envs = feet_pos.shape[0]
    feet_angles = torch.zeros(num_envs, 12, device="cuda", dtype=torch.float)
    for i in range(4):
        feet_angles[:, 3*i:3*i+3] = foot_pos_in_hip_frame_to_joint_angle(feet_pos[:, 3*i:3*i+3],
                                                       l_hip_sign=(-1) ** (i + 1))
    return feet_angles  # torch.Size([num_envs, 12])


def angles_in_hip_frame_to_feet_pos_np(joint_angles):
    """
    Description: Calculate foot pos of four legs based on the given joint angles
    -----------
    Input: joint angles of four legs (FR/FL/RR/RL) -> torch.Tensor Shape([num_envs, 12])
    Output: foot position of four legs (FR/FL/RR/RL) -> torch.Tensor Shape([num_envs, 12])
    """
    num_envs = joint_angles.shape[0]
    feet_pos = np.zeros([num_envs, 12])
    for i in range(4):
        feet_pos[:, 3 * i:3 * i + 3] = foot_position_in_hip_frame(joint_angles[:, 3 * i:3 * i + 3], l_hip_sign=(-1) ** (i + 1))
    feet_pos = np.array(feet_pos)
    return feet_pos

# def quat2euler(quat):
#     """
#     Input: Quaternion: torch.Size([num_envs, 4]); (x, y, z, w)
#     Output: Euler angles (degree): torch.Size([num_envs, 3]); Roll(x), Pitch(y), Yaw(z)
#     """
#     qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
#     roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
#     pitch = torch.asin(2 * (qw * qy - qz * qx))
#     yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
#     return torch.cat((roll, pitch, yaw), dim=-1)


def quat2euler(quat):
    """
    Input:
        Quaternion: torch.Size([num_envs, 4]); (x, y, z, w)
    Output:
        Euler angles (rad): torch.Size([num_envs, 3]); Roll(x), Pitch(y), Yaw(z)
    """
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = torch.asin(2 * (qw * qy - qz * qx))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return torch.stack((roll, pitch, yaw), dim=-1)


def euler2quat(euler_angles):
    """
    将欧拉角转换为四元数

    参数:
        euler_angles (torch.Tensor): 欧拉角张量，形状为 (num_envs, 3)，第二维度为 (roll, pitch, yaw)

    返回:
        torch.Tensor: 四元数张量，形状为 (num_envs, 4)，第二维度为 (x, y, z, w)
    """
    roll, pitch, yaw = euler_angles.unbind(dim=1)

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qx, qy, qz, qw], dim=1)

    return quaternion


def euler2matrix(euler_angles):
    """ 将欧拉角批量转换为旋转矩阵，使用 ZYX（yaw-pitch-roll）的旋转顺序
    :param euler_angles: 形状为 (batch_size, 3) 的欧拉角张量，每行的三个变量分别对应横滚角 roll、俯仰角 pitch 和偏航角 yaw（rad）
    :return: 形状为 (batch_size, 3, 3) 的旋转矩阵张量
    """
    roll, pitch, yaw = euler_angles.unbind(dim=1)

    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    batch_size = euler_angles.size(0)

    R_x = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device="cuda")
    R_x[:, 0, 0] = 1
    R_x[:, 1, 1] = cos_roll
    R_x[:, 1, 2] = -sin_roll
    R_x[:, 2, 1] = sin_roll
    R_x[:, 2, 2] = cos_roll

    R_y = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device="cuda")
    R_y[:, 0, 0] = cos_pitch
    R_y[:, 0, 2] = sin_pitch
    R_y[:, 1, 1] = 1
    R_y[:, 2, 0] = -sin_pitch
    R_y[:, 2, 2] = cos_pitch

    R_z = torch.zeros((batch_size, 3, 3), dtype=torch.float32, device="cuda")
    R_z[:, 0, 0] = cos_yaw
    R_z[:, 0, 1] = -sin_yaw
    R_z[:, 1, 0] = sin_yaw
    R_z[:, 1, 1] = cos_yaw
    R_z[:, 2, 2] = 1

    # 合并旋转
    rotation_matrix = torch.bmm(R_z, torch.bmm(R_y, R_x))

    return rotation_matrix


def quat2matrix(quaternions):
    x, y, z, w = quaternions.unbind(dim=1)

    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w

    yy = y * y
    yz = y * z
    yw = y * w

    zz = z * z
    zw = z * w

    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
    ], dim=-1).view(-1, 3, 3)

    return rotation_matrix


def quaternion_distance(q1, q2):
    # Calculate the dot product between the normalized quaternions
    dot_product = torch.sum(q1 * q2, dim=1)

    # Calculate the angle between the quaternions
    angle = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    return angle