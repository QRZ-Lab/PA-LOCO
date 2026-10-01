# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""
2023.12.1:
    添加函数 _init_, randomized_terrain, curriculum, selected_terrain, make_terrain, add_terrain_to_map 的注释
"""

import numpy as np
from numpy.random import choice

from isaacgym import terrain_utils
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.utils import terrain_utils_addition


class Terrain:
    def __init__(self, cfg: BaseConfig, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots

        # 如果地形类型是 plane 或者 none, 则不进行初始化
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        # 从 config 中获取环境长宽并获取不同环境生成比例
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # 计算子地形数量
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols  # 10 * 20
        # 储存环境原点信息
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # 边界大小
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # 总行数和列数
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 用于存储地形的高度信息，初始值全为零
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        # 根据配置参数设置调用不同地形的生成方式
        if cfg.curriculum:  # 地形课程学习的方式
            self.curriculum()
        elif cfg.selected:  # 选择特定地形
            self.selected_terrain()
        else:  # 根据 self.proportions 随机生成
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        # 将地形进行转化
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_threshold)
    
    def randomized_terrain(self):
        """ 子区域生成随机地形。通过选择地形类型（choice）和难度级别（difficulty）引入了随机性
        :return: None
        """
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            # 索引 k 被转换为相应的行和列索引 (i, j)
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # choice 采样至 [0, 1] 均匀分布, 用于确定要生成的地形类型
            # difficulty 被赋予一个从列表中随机选择的值。这个值表示地形的难度
            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])  # 0.5  # testing: slope = diff*0.4 = 0.2
            difficulty = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # 0.5  # testing: slope = diff*0.4 = 0.2

            # 生成特定类型的地形并使用行列索引将其添加到整体的地形图中
            if hasattr(self.cfg, 'platform_size'):
                terrain = self.make_terrain(choice, difficulty, platform_size=self.cfg.platform_size)
            else:
                terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curriculum(self):
        """ 按照课程学习的方式生成地形，通过逐行增加的难度级别和逐列增加的选择值(地形类型)，调整生成的地形，然后将其添加到地图的相应位置
        :return: None
        """
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows  # 难度级别随着行数增加而增加
                choice = j / self.cfg.num_cols + 0.001

                # 生成特定类型的地形并使用行列索引将其添加到整体的地形图中
                if hasattr(self.cfg, 'platform_size'):
                    terrain = self.make_terrain(choice, difficulty, platform_size=self.cfg.platform_size)
                else:
                    terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        """ 根据配置中指定的地形类型和其他参数，使用循环为每个子地形生成特定类型的地形，并将其添加到地图的相应位置
        :return: None
        """
        # 通过 terrain_kwargs 获取配置中指定的地形类型
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            # 索引 k 被转换为相应的行和列索引 (i, j)
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=self.width_per_env_pixels,
                                               length=self.width_per_env_pixels,
                                               vertical_scale=self.cfg.vertical_scale,
                                               horizontal_scale=self.cfg.horizontal_scale)

            # 根据 terrain_type 动态调用地形生成函数
            # 假设 terrain_type 对应于某个地形生成函数的名称。额外的参数存储在 self.cfg.terrain_kwargs.terrain_kwargs 中
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty, platform_size=3):
        """
        根据给定的选择值和难度级别生成具体类型的地形，并返回相应的 SubTerrain 实例
        :param choice: 随机数，用于选择生成的地形类型 -> np.random.uniform(0, 1)
        :param difficulty: 难度
        :return: SubTerrain
        """
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=self.width_per_env_pixels,
                                           length=self.width_per_env_pixels,
                                           vertical_scale=self.cfg.vertical_scale,
                                           horizontal_scale=self.cfg.horizontal_scale)
        # 通过难度调整坡度
        slope = difficulty * 0.4
        # slope = difficulty * 0.75

        step_height = 0.05 + 0.18 * difficulty
        step_width = np.random.uniform(0.25, 0.35, 1)
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        discrete_stairs_height = 0.02 + 0.08 * difficulty
        # 注意 self.proportions 的元素之和应为 1, 由 choice 的值决定生成地形的类型
        if choice < self.proportions[0]:
            # 锥形坡地形, 坡度可能为正或负
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)  # platform_size=3
        elif choice < self.proportions[1]:
            # 生成带坑洼的锥形坡地形
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=platform_size)  # platform_size=3
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            # 生成金字塔台阶地形, step_height 由 difficulty 进行调节
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=platform_size)
        elif choice < self.proportions[4]:
            # 生成离散障碍地形, discrete_obstacles_height 由 difficulty 进行调节
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=platform_size)
        elif choice < self.proportions[5]:
            # 生成梅花桩地形, stepping_stones_size 和 stone_distance 由 difficulty 进行调节
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            # 生成存在 gap 的地形, gap_size 由 difficulty 进行调节
            terrain_utils_addition.gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            # 生成深坑地形, pit_depth 由 difficulty 进行调节
            terrain_utils_addition.pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        else:
            # terrain_utils_addition.enclosure_terrain(terrain)  # 生成柱子地形
            # terrain_utils_addition.random_pillars_terrain(terrain, num_pillars=20, pillar_length=0.2, pillar_width=0.2, pillar_height=1.0, platform_size=3.)
            # terrain_utils_addition.random_noise_terrain(terrain)  # 随机噪声地形
            terrain_utils_addition.random_discrete_stair_terrain(terrain, random_height_amp=discrete_stairs_height, platform_size=platform_size)  # platform_size=3
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        """ 将生成的地形添加到整体地形图中，并更新相关的环境原点信息
        :param terrain: 生成的子地形
        :param row: 子地形在整体地形中的行索引
        :param col: 子地形在整体地形中的列索引
        :return: None
        """
        i = row
        j = col
        # map coordinate system
        # 通过所生成地形的行列索引生成地形在整体地形图中的起始和结束坐标，这里考虑了 border
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        # 在相应位置更新整体地形图的高度信息
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        # 计算子地形的环境原点信息
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
