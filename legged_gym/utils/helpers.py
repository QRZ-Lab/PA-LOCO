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
Logs
----
2023.12.27:
    函数 export_policy_as_jit 中新增将含 encoder 的网络进行导出的功能
    新增类 PolicyExporterEncoder

2023.12.28:
    新增函数 get_load_run_dir_path, 用于获取 'load_run' 的路径
"""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from rsl_rl.utils.tcn_encoder import TcnEncoder  # TODO


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    runs = [os.path.join(root, name) for name in os.listdir(root)
            if name != 'exported' and os.path.isdir(os.path.join(root, name))]
    if not runs:
        raise ValueError("No runs in this directory: " + root)
    last_run = max(runs, key=os.path.getmtime)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if file.startswith('model_') and file.endswith('.pt')]
        if not models:
            raise ValueError("No models in this directory: " + load_run)
        model = max(models, key=lambda name: int(name[6:-3]))
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def get_load_run_dir_path(root, load_run=-1):
    """ 返回 load_run 的路径，以及对应的字符串名 """
    runs = [name for name in os.listdir(root)
            if name != 'exported' and os.path.isdir(os.path.join(root, name))]
    if not runs:
        raise ValueError("No runs in this directory: " + root)
    last_run_name = max(runs, key=lambda name: os.path.getmtime(os.path.join(root, name)))
    last_run = os.path.join(root, last_run_name)
    # 如果 load_run 被设置为 -1，则使用排序后列表中的最后一个 run 的路径; 否则使用指定的 run 的路径
    if load_run == -1:
        load_run = last_run
        load_run_name = last_run_name
    else:
        load_run_name = load_run
        load_run = os.path.join(root, load_run)
    return load_run, load_run_name


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "force_sep_t", "help": "Training task name."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--teacher_checkpoint", "type": str, "help": "Teacher checkpoint for residual training."},
        {"name": "--student_checkpoint", "type": str, "help": "Student checkpoint for residual training."},
        {"name": "--student_policy", "type": str, "help": "Student TorchScript policy for residual training."},
        {"name": "--residual_checkpoint", "type": str, "help": "Residual checkpoint for simulation rollout."},
        {"name": "--steps", "type": int, "default": 1000, "help": "Simulation rollout steps."},
        {"name": "--mefa", "action": "store_true", "default": False, "help": "Run the force_sep student without the residual policy."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name alignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


# %%%%%% Policy Exporter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def export_policy_as_jit(actor_critic, path, **kwargs):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterGRU(actor_critic)
        # exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    elif hasattr(actor_critic, 'gating_network'):
        if hasattr(actor_critic, 'is_long_short_encoder'):
            exporter = PolicyExporterGatingLongShortEncoder(actor_critic)
            exporter.export(path)
    elif hasattr(actor_critic, 'residual'):
        exporter = PolicyExporterMlpEncoderResidual(actor_critic)
        exporter.export(path)
    elif hasattr(actor_critic, 'encoder'):
        if hasattr(actor_critic, 'is_simple_tcn_encoder'):
            exporter = PolicyExporterSimpleTcnEncoder(actor_critic)
            exporter.export(path)
        if hasattr(actor_critic, 'is_mlp_encoder'):
            exporter = PolicyExporterMlpEncoder(actor_critic)
            exporter.export(path)
        if hasattr(actor_critic, 'is_tcn_encoder'):
            exporter = PolicyExporterTcnEncoder(actor_critic)
            exporter.export(path)  # TODO
    elif hasattr(actor_critic, 'long_hist_encoder'):
        if hasattr(actor_critic, 'is_long_short_encoder'):
            exporter = PolicyExporterMlpLongShortEncoder(actor_critic)
            exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterTcnEncoder(torch.nn.Module):  # TODO
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        # Create a new instance of custom TCN module
        encoder_layers = [
            TcnEncoder(num_inputs=actor_critic.num_base_obs, history_length=actor_critic.history_length,
                       num_outputs=actor_critic.num_latency, num_channels=actor_critic.tcn_channels,
                       kernel_size=actor_critic.tcn_kernel_size, dropout=actor_critic.tcn_dropout)
        ]

        self.tcn_encoder = nn.Sequential(*encoder_layers)
        self.tcn_encoder.load_state_dict(actor_critic.encoder.state_dict())  # Load parameters from the original encoder

    def forward(self, x):
        # x.shape: batch_size, num_obs, history_length
        latent = self.tcn_encoder(x)
        base_obs = x[:, :, -1].detach().clone()
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        return self.actor(actor_input_obs), latent

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterMlpEncoder(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.mlp_encoder = copy.deepcopy(actor_critic.encoder)
        self.num_obs = actor_critic.num_base_obs

    def forward(self, x):
        # x.shape: (batch_size, num_obs * history_length)
        base_obs = x[:, -self.num_obs:].detach().clone()
        latent = self.mlp_encoder(x)
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        return self.actor(actor_input_obs), latent

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterMlpLongShortEncoder(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.long_hist_encoder = copy.deepcopy(actor_critic.long_hist_encoder)
        self.short_hist_encoder = copy.deepcopy(actor_critic.short_hist_encoder)
        self.num_obs = actor_critic.num_base_obs

    def forward(self, x):
        # x.shape: (batch_size, num_obs * history_length)
        base_obs = x[:, -self.num_obs:].detach().clone()
        latent = torch.cat((self.long_hist_encoder(x), self.short_hist_encoder(x[:, -x.shape[1]//2:])), dim=-1)
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        return self.actor(actor_input_obs), latent

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class PolicyExporterGatingLongShortEncoder(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.gating_network = copy.deepcopy(actor_critic.gating_network)

        self.quad_actor = copy.deepcopy(actor_critic.expert_quad)
        self.quad_long_hist_encoder = copy.deepcopy(actor_critic.expert_quad_long_hist_encoder)
        self.quad_short_hist_encoder = copy.deepcopy(actor_critic.expert_quad_short_hist_encoder)

        self.tripod_actor = copy.deepcopy(actor_critic.expert_tripod)
        self.tripod_long_hist_encoder = copy.deepcopy(actor_critic.expert_tripod_long_hist_encoder)
        self.tripod_short_hist_encoder = copy.deepcopy(actor_critic.expert_tripod_short_hist_encoder)

        self.fall_recovery_actor = copy.deepcopy(actor_critic.expert_fall_recovery)
        self.fall_recovery_long_hist_encoder = copy.deepcopy(actor_critic.expert_fall_recovery_long_hist_encoder)
        self.fall_recovery_short_hist_encoder = copy.deepcopy(actor_critic.expert_fall_recovery_short_hist_encoder)

        self.num_obs = actor_critic.num_base_obs
        self.num_gating_obs = actor_critic.num_gating_obs
        self.num_history = actor_critic.num_history
        self.num_cmd_modes = actor_critic.num_cmd_modes

    def forward(self, x):
        # x.shape: (batch_size, num_obs * history_length)
        gating_obs = x[:, -self.num_gating_obs:].detach().clone()
        base_obs = gating_obs[:, -self.num_obs:]

        expert_selection = self.gating_network(gating_obs)

        # 移除 mode cmd (obs history for expert) 注意使用 .view() 是各个维度的长度
        observations = x.view(x.shape[0], self.num_history, self.num_gating_obs)[:, :, self.num_cmd_modes:].reshape(x.shape[0], -1)

        quad_latent = torch.cat((
            self.quad_long_hist_encoder(observations),
            self.quad_short_hist_encoder(observations[:, -observations.shape[1] // 2:])), dim=-1)
        tripod_latent = torch.cat((
            self.tripod_long_hist_encoder(observations),
            self.tripod_short_hist_encoder(observations[:, -observations.shape[1] // 2:])), dim=-1)
        fall_recovery_latent = torch.cat((
            self.fall_recovery_long_hist_encoder(observations),
            self.fall_recovery_short_hist_encoder(observations[:, -observations.shape[1] // 2:])), dim=-1)

        quad_mean = self.quad_actor(torch.cat((base_obs, quad_latent), dim=-1))
        tripod_mean = self.tripod_actor(torch.cat((base_obs, tripod_latent), dim=-1))
        fall_recovery_mean = self.fall_recovery_actor(torch.cat((base_obs, fall_recovery_latent), dim=-1))

        mean = expert_selection[:, 0].unsqueeze(dim=-1) * quad_mean + \
               expert_selection[:, 1].unsqueeze(dim=-1) * tripod_mean + \
               expert_selection[:, 2].unsqueeze(dim=-1) * fall_recovery_mean
        return mean, expert_selection

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterMlpEncoderResidual(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        # self.actor = copy.deepcopy(actor_critic.actor)
        # self.mlp_encoder = copy.deepcopy(actor_critic.encoder)
        self.residual = copy.deepcopy(actor_critic.residual)
        self.num_obs = actor_critic.num_base_obs

    def forward(self, x):
        # x.shape: (batch_size, num_obs * history_length)
        # base_obs = x[:, -self.num_obs:].detach().clone()
        # latent = self.mlp_encoder(x)
        # actor_obs = torch.cat((base_obs, latent), dim=-1)
        # actions_mean = self.actor(actor_obs)
        actions_mean_residual = self.residual(x)
        return actions_mean_residual

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterSimpleTcnEncoder(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.simple_tcn_encoder = copy.deepcopy(actor_critic.encoder)
        self.num_obs = actor_critic.num_base_obs
        self.history_length = actor_critic.history_length

    def forward(self, x):
        # x.shape: (batch_size, num_obs * history_length)
        base_obs = x[:, -self.num_obs:].detach().clone()
        latent = self.simple_tcn_encoder(x.reshape(x.size(0), base_obs.size(1), -1).permute(0, 2, 1))
        actor_input_obs = torch.cat((base_obs, latent), dim=-1)
        return self.actor(actor_input_obs), latent

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterGRU(torch.nn.Module):  # TODO: check
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, h = self.memory(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class PolicyExporterGRUMlpEncoder(torch.nn.Module):  # TODO: check
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        # input shape: batch_size, num_base_obs * history_length
        base_obs = x[:, -self.num_base_obs:].detach().clone()  # input shape: batch_size, num_base_obs
        latent = self.encoder(x)  # input shape: batch_size, num_base_obs * history_length
        memory_obs = torch.cat((base_obs, latent), dim=-1)
        out, h = self.memory(memory_obs.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        return self.actor(out.squeeze(0)), latent

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
