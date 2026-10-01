import numpy as np

"""
2023.12.01:
    legged_gym.utils.terrain.py 用于生成地形，其中各种不同地形从 isaacgym 的 terrain_utils 中 import
    legged_gym.utils.terrain_utils_addition.py 用于补充 isaacgym terrain_utils 中不存在的一些新地形
    增加函数 gap_terrain, pit_terrain 的函数注释
    
    添加函数 random_pillars_terrain, 用于随机生成柱子的地形
    添加函数 enclosure_terrain, 用于生成含有围墙和障碍物的地形
"""


def gap_terrain(terrain, gap_size, platform_size=1.):
    """
    :param terrain: 地形
    :param gap_size: 缺口尺寸 [meter]
    :param platform_size: 地形中心的平地尺寸 [meter]
    :return: SubTerrain 的实例
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 地形中心点位置
    center_x = terrain.length // 2
    center_y = terrain.width // 2
    # 缺口起止点
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    # 在地形中创建了一个围绕地形中心平台且深度为 -1000 的环形缺口
    terrain.height_field_raw[center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0
    return terrain


def pit_terrain(terrain, depth, platform_size=1.):
    """
    在地形中心创建一个深坑，深坑的深度和平地区域的尺寸可以通过参数进行调整

    :param terrain: 地形
    :param depth: 深度
    :param platform_size: 地形中心的平地尺寸
    :return: terrain (SubTerrain): 更新后的地形
    """
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    # 平地高度下沉
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    return terrain


def random_pillars_terrain(terrain, num_pillars, pillar_length, pillar_width, pillar_height=1., platform_size=1):
    """
    生成带有随机柱子的地形

    :param terrain: 地形
    :param num_pillars: 柱子数量
    :param pillar_length: 每个柱子的长度
    :param pillar_width: 每个柱子的宽度
    :param pillar_height: 每个柱子的高度
    :param platform_size: 中心平台的大小
    :return: terrain (SubTerrain): 更新后的地形
    """
    # 将参数转换为离散单位
    pillar_length = int(pillar_length / terrain.horizontal_scale)
    pillar_width = int(pillar_width / terrain.horizontal_scale)
    pillar_height = int(pillar_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 在地形中心添加一个平坦的平台
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # 计数器，用于计算成功生成的柱子数量
    success_counter = 0
    while success_counter < num_pillars:
        # 在每个区域内随机放置一个柱子
        pillar_start_x = np.random.randint(0, terrain.width - pillar_length)
        pillar_start_y = np.random.randint(0, terrain.length - pillar_width)
        stop_x = pillar_start_x + pillar_length
        stop_y = pillar_start_y + pillar_width
        # 检查柱子长宽所形成的长方形与平台所形成的长方形是否有重叠部分
        overlap = not (stop_x < x1 or pillar_start_x > x2 or stop_y < y1 or pillar_start_y > y2)
        if overlap:
            # 如果柱子与平台重叠，则跳过此次迭代
            continue
        terrain.height_field_raw[pillar_start_x: stop_x, pillar_start_y: stop_y] = pillar_height
        success_counter += 1

    return terrain


def enclosure_terrain(terrain, wall_width=0.2, wall_height=1, obs_length=2, obs_width=0.3, obs_height=0.75, platform_size=1):
    """
    生成带有围墙和障碍物的地形。

    :param terrain: 地形
    :param wall_width: 围墙宽度
    :param wall_height: 围墙高度
    :param obs_length: 障碍物长度
    :param obs_height: 障碍物高度
    :param obs_width: 障碍物宽度
    :param platform_size: 地形中心的平台的尺寸
    :return: terrain (SubTerrain): 更新后的地形
    """
    # 将参数转换为离散单位
    wall_width = int(wall_width / terrain.horizontal_scale)
    wall_height = int(wall_height / terrain.vertical_scale)
    obs_length = int(obs_length / terrain.horizontal_scale)
    obs_width = int(obs_width / terrain.horizontal_scale)
    obs_height = int(obs_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 生成四周围墙
    terrain.height_field_raw[:wall_width, :] = wall_height
    terrain.height_field_raw[-wall_width:, :] = wall_height
    terrain.height_field_raw[:, :wall_width] = wall_height
    terrain.height_field_raw[:, -wall_width:] = wall_height

    # 在地形中心生成平台
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # 分割区域
    rows = np.linspace(0, terrain.width, 15, dtype=int)
    cols = np.linspace(0, terrain.length, 15, dtype=int)
    # 选择障碍物起始点，并存储在数组中
    obs_start_pos = [(rows[2], cols[3]), (rows[8], cols[11]),
                     (rows[2], cols[11]), (rows[8], cols[3]),
                     (rows[5], cols[9])]
    # 生成障碍物
    for point in obs_start_pos:
        # 障碍物的四个顶点坐标
        obs_top_left = (point[0], point[1])
        obs_bottom_right = (point[0] + obs_length, point[1] + obs_width)
        # 检查障碍物和平台是否有重叠部分
        overlap = not (obs_bottom_right[0] < x1 or obs_bottom_right[1] < y1 or
                       obs_top_left[0] > x1 or obs_top_left[1] > y2)
        if overlap:
            print(f"Warning: Some obstacle overlaps with the platform and cannot be placed.")
        else:
            # 在选择的点放置障碍物
            terrain.height_field_raw[point[0]:point[0] + obs_length, point[1]:point[1] + obs_width] = obs_height
    return terrain


def random_noise_terrain(terrain, random_height_amp=0.05, platform_size=2):
    """
    生成随机噪声的地形
    """
    # 将参数转换为离散单位
    random_height_amp = int(random_height_amp / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 在地形中心添加一个平坦的平台
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # 生成随机离散阶梯
    for i in range(terrain.width):
        for j in range(terrain.length):
            # 检查当前位置是否在平台上，如果是则跳过
            if x1 <= i < x2 and y1 <= j < y2:
                continue
            # 从均匀分布中采样高度
            height = np.random.uniform(low=-random_height_amp, high=random_height_amp)
            terrain.height_field_raw[i, j] = height

    return terrain


def random_discrete_stair_terrain(terrain, grid_length=0.35, grid_width=0.35, random_height_amp=0.05, platform_size=0.5):
    """
    生成随机噪声的地形
    """
    # 将参数转换为离散单位
    grid_length = int(grid_length / terrain.horizontal_scale)
    grid_width = int(grid_width / terrain.horizontal_scale)
    random_height_amp = int(random_height_amp / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    # 网格数目
    num_grid_width = int(terrain.width / grid_width)
    num_grid_length = int(terrain.length / grid_length)

    # 在地形中心添加一个平坦的平台
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    x_start = 0
    # 生成随机离散阶梯
    for i in range(num_grid_width):
        x_end = x_start + grid_width
        y_start = 0
        for j in range(num_grid_length):
            y_end = y_start + grid_length
            height = np.random.uniform(low=-random_height_amp, high=random_height_amp)  # 从均匀分布中采样高度
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = height
            y_start = y_end
        x_start = x_end

    terrain.height_field_raw[x1:x2, y1:y2] = 0

    return terrain


def template_terrain(terrain):
    """ 生成地形
    :param
    :return: SubTerrain 的实例
    """
    return terrain