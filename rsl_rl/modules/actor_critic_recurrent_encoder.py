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
from .actor_critic_encoder import ActorCriticMlpEncoder, get_activation
from rsl_rl.utils import unpad_trajectories


class ActorCriticRecurrentMlpEncoder(ActorCriticMlpEncoder):
    is_recurrent = True
    is_mlp_encoder = True

    def __init__(self, env_cfg,
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 actor_hidden_dims=None,
                 critic_hidden_dims=None,
                 activation='elu',
                 rnn_type='lstm',
                 rnn_hidden_size=256,
                 rnn_num_layers=1,
                 init_noise_std=1.0,
                 is_teacher=True,
                 mlp_input_dim_e_t=None,
                 mlp_output_dim_e_t=None,
                 mlp_hidden_dims_e_t=None,
                 mlp_input_dim_e_s=None,
                 mlp_output_dim_e_s=None,
                 mlp_hidden_dims_e_s=None,
                 **kwargs):
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256, 256]
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 256, 256]
        if mlp_hidden_dims_e_t is None:
            mlp_hidden_dims_e_t = [256, 128]
        if mlp_hidden_dims_e_s is None:
            mlp_hidden_dims_e_s = [1024, 512, 256, 128]
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(env_cfg=env_cfg,
                         num_actor_obs=rnn_hidden_size,
                         num_critic_obs=rnn_hidden_size,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         is_teacher=is_teacher,
                         mlp_input_dim_e_t=mlp_input_dim_e_t,
                         mlp_output_dim_e_t=mlp_output_dim_e_t,
                         mlp_hidden_dims_e_t=mlp_hidden_dims_e_t,
                         mlp_input_dim_e_s=mlp_input_dim_e_s,
                         mlp_output_dim_e_s=mlp_output_dim_e_s,
                         mlp_hidden_dims_e_s=mlp_hidden_dims_e_s,
                         )

        activation = get_activation(activation)

        self.memory_input_dim_a = env_cfg["env"]["num_base_obs"] + env_cfg["env"]["num_latent"]
        self.memory_input_dim_c = num_critic_obs

        self.memory_a = Memory(self.memory_input_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(self.memory_input_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        """ 更新分布返回采样后的动作 """
        if self.is_teacher:
            return self.act_teacher(observations, masks=masks, hidden_states=hidden_states, **kwargs)
        else:
            return self.act_student(observations, **kwargs)

    def act_inference(self, observations, masks=None, hidden_states=None, **kwargs):
        """ 策略推理 """
        if self.is_teacher:
            return self.act_inference_teacher(observations, **kwargs)
        else:
            return self.act_inference_student(observations, **kwargs)

    def act_teacher(self, observations, masks=None, hidden_states=None, **kwargs):
        base_obs = self.get_base_obs(observations)
        height_obs = self.get_height_obs(observations)
        extrinsic_obs = self.get_extrinsic_obs(observations)
        latent = self.encoder(torch.cat((height_obs, extrinsic_obs), dim=-1))
        memory_a_obs = torch.cat((base_obs, latent), dim=-1)
        input_a = self.memory_a(memory_a_obs, masks, hidden_states)
        return super().act_teacher(input_a.squeeze(0))

    def act_student(self, observations, **kwargs):
        base_obs = observations[:, -self.num_base_obs:]  # input shape: batch_size, num_base_obs
        latent = self.encoder(observations)  # input shape: batch_size, num_base_obs * history_length
        memory_a_obs = torch.cat((base_obs, latent), dim=-1)
        input_a = self.memory_a(memory_a_obs)
        return super().act_student(input_a.squeeze(0))

    def act_inference_teacher(self, observations, **kwargs):
        base_obs = self.get_base_obs(observations)
        height_obs = self.get_height_obs(observations)
        extrinsic_obs = self.get_extrinsic_obs(observations)
        latent = self.encoder(torch.cat((height_obs, extrinsic_obs), dim=-1))
        memory_a_obs = torch.cat((base_obs, latent), dim=-1)
        input_a = self.memory_a(memory_a_obs)
        return super().act_inference_teacher(input_a.squeeze(0))

    def act_inference_student(self, observations, **kwargs):
        base_obs = observations[:, -self.num_base_obs:]  # input shape: batch_size, num_base_obs
        latent = self.encoder(observations)  # input shape: batch_size, num_base_obs * history_length
        memory_a_obs = torch.cat((base_obs, latent), dim=-1)
        input_a = self.memory_a(memory_a_obs)
        return super().act_inference_student(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0