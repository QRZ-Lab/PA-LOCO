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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCriticMlpEncoderResidual(nn.Module):
    is_recurrent = False
    is_mlp_encoder = True

    def __init__(self,
                 env_cfg,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=None,
                 residual_hidden_dims=None,
                 critic_hidden_dims=None,
                 activation='elu',
                 init_noise_std=1.0,
                 is_teacher=True,
                 encoder_output_tanh=False,
                 mlp_input_dim_e_t=None,
                 mlp_output_dim_e_t=None,
                 mlp_hidden_dims_e_t=None,
                 mlp_input_dim_e_s=None,
                 mlp_output_dim_e_s=None,
                 mlp_hidden_dims_e_s=None,
                 **kwargs):
        self.res_action_mean = None
        self.main_action_mean = None
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
        if residual_hidden_dims is None:
            residual_hidden_dims = [256, 128, 64]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 256]
        if mlp_hidden_dims_e_t is None:
            mlp_hidden_dims_e_t = [256, 128]
        if mlp_hidden_dims_e_s is None:
            mlp_hidden_dims_e_s = [1024, 512, 256, 128]
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticMlpEncoderResidual, self).__init__()

        activation = get_activation(activation)  # 选择配置文件所指定的 activation function
        
        self.is_teacher = is_teacher
        self.encoder_output_tanh = encoder_output_tanh
        self.num_base_obs = num_base_obs = env_cfg["env"]["num_base_obs"]
        self.num_height_obs = num_height_obs = env_cfg["env"]["num_height_obs"]
        self.num_extrinsic_obs = num_extrinsic_obs = env_cfg["env"]["num_extrinsic_obs"]
        self.num_residual_obs = num_residual_obs = env_cfg["env"]["num_residual_obs"]
        
        self.get_base_obs = lambda obs: obs[..., :num_base_obs]
        self.get_height_obs = lambda obs: obs[..., num_base_obs:num_base_obs+num_height_obs]
        self.get_extrinsic_obs = lambda obs: obs[..., num_base_obs+num_height_obs:num_base_obs+num_height_obs+num_extrinsic_obs]

        # %%%%%% Encoder network: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        if self.is_teacher:
            mlp_input_dim_e, mlp_hidden_dims_e, mlp_output_dim_e = mlp_input_dim_e_t, mlp_hidden_dims_e_t, mlp_output_dim_e_t
        else:
            mlp_input_dim_e, mlp_hidden_dims_e, mlp_output_dim_e = mlp_input_dim_e_s, mlp_hidden_dims_e_s, mlp_output_dim_e_s

        encoder_layers = [nn.Linear(mlp_input_dim_e, mlp_hidden_dims_e[0]), activation]
        for l in range(len(mlp_hidden_dims_e)):
            if l == len(mlp_hidden_dims_e) - 1:
                encoder_layers.append(nn.Linear(mlp_hidden_dims_e[l], mlp_output_dim_e))
                if self.encoder_output_tanh:
                    encoder_layers.append(nn.Tanh())
            else:
                encoder_layers.append(nn.Linear(mlp_hidden_dims_e[l], mlp_hidden_dims_e[l + 1]))
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        mlp_input_dim_a = env_cfg["env"]["num_base_obs"] + env_cfg["env"]["num_latent"]
        mlp_input_dim_r = num_residual_obs
        mlp_input_dim_c = num_critic_obs

        # %%%%%% Policy network: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        actor_layers = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), activation]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        # 将 actor_layers 中的层按照顺序添加到 self.actor 中，构建了整个 Actor 网络的前向传播过程
        self.actor = nn.Sequential(*actor_layers)

        # %%%%%% Residual policy network: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        residual_layers = [nn.Linear(mlp_input_dim_r, residual_hidden_dims[0]), activation]
        for l in range(len(residual_hidden_dims)):
            if l == len(residual_hidden_dims) - 1:
                residual_layers.append(nn.Linear(residual_hidden_dims[l], num_actions))
            else:
                residual_layers.append(nn.Linear(residual_hidden_dims[l], residual_hidden_dims[l + 1]))
                residual_layers.append(activation)
        # 将 residual_layers 中的层按照顺序添加到 self.residual 中，构建了整个 residual 网络的前向传播过程
        self.residual = nn.Sequential(*residual_layers)

        # %%%%%% Value function: MLP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Residual MLP: {self.residual}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        """ 计算熵 """
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """ 获取策略网络的输出动作的分布 """
        # mean = self.actor(observations)
        mean_residual = self.residual(observations)
        # mean = mean + mean_residual
        self.distribution = Normal(mean_residual, mean_residual*0. + self.std)

    def get_actions_log_prob(self, actions):
        """ 计算给定动作 actions 的对数概率 """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act(self, observations, **kwargs):
        """ 从更新后的策略分布中采样动作得到具体的动作 """
        if self.is_teacher:
            return self.act_teacher(observations, **kwargs)
        else:
            return self.act_student(observations, **kwargs)

    def act_inference(self, observations, **kwargs):
        """ 策略推理 """
        if self.is_teacher:
            return self.act_inference_teacher(observations, **kwargs)
        else:
            return self.act_inference_student(observations, **kwargs)

    def get_actor_obs(self, observations):
        if self.is_teacher:
            return self._get_actor_obs_teacher(observations)
        else:
            return self._get_actor_obs_student(observations)

    def evaluate(self, critic_observations, **kwargs):
        """ 计算状态值函数 """
        value = self.critic(critic_observations)
        return value

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def act_teacher(self, observations, **kwargs):
        """ 执行 teacher 网络输出的动作 """
        base_obs = self.get_base_obs(observations)
        height_obs = self.get_height_obs(observations)
        extrinsic_obs = self.get_extrinsic_obs(observations)

        latent = self.encoder(torch.cat((height_obs, extrinsic_obs), dim=-1))
        actor_obs = torch.cat((base_obs, latent), dim=-1)
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_student(self, observations, **kwargs):
        """ 执行 student 网络输出的动作 """
        # base_obs = observations[:, -self.num_base_obs:]  # input shape: batch_size, num_base_obs
        # latent = self.encoder(observations)  # input shape: batch_size, num_base_obs * history_length
        # actor_obs = torch.cat((base_obs, latent), dim=-1)

        self.update_distribution(observations)  # actor_obs+latent
        return self.distribution.sample()

    def act_inference_student(self, observations, **kwargs):
        """ 执行 student 网络输出的动作 """
        # base_obs = observations[:, -self.num_base_obs:]  # input shape: batch_size, num_base_obs
        # latent = self.encoder(observations)  # input shape: batch_size, num_base_obs * history_length
        # actor_obs = torch.cat((base_obs, latent), dim=-1)
        # actions_mean = self.actor(actor_obs)
        actions_mean_residual = self.residual(observations)
        return actions_mean_residual

    def _get_actor_obs_student(self, observations):
        base_obs = observations[:, -self.num_base_obs:]  # input shape: batch_size, num_base_obs
        latent = self.encoder(observations)  # input shape: batch_size, num_base_obs * history_length
        return torch.cat((base_obs, latent), dim=-1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
