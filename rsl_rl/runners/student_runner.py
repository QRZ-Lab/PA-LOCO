"""
Logs
----

2023.12.26:
    增加函数 _copy_base_policy_params
"""

import os
import time
import math
import json
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.modules import ActorCriticMlpEncoder, ActorCriticTcnEncoder
from rsl_rl.env import VecEnv
from rsl_rl.utils.utils import create_folders


class StudentRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 env_cfg,
                 teacher,
                 student,
                 log_dir=None,
                 learning_rate=2e-4,
                 device='cpu',
                 skip_reset=False):

        self.cfg = train_cfg["runner"]
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.device = device
        self.env = env

        # 初始化一些训练过程中需要的参数
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # 指定 teacher 网络和 student 网络以及优化器
        self.teacher: ActorCriticMlpEncoder = teacher
        self.student: ActorCriticTcnEncoder = student

        # 学习率设置 TODO
        self.lr = self.train_cfg["runner"]["learning_rate"] if 'learning_rate' in self.train_cfg[
            "runner"] else learning_rate
        self.exp_decay_a = self.train_cfg["runner"]["exp_decay_a"] if 'exp_decay_a' in self.train_cfg["runner"] else 1
        self.exp_decay_b = self.train_cfg["runner"]["exp_decay_b"] if 'exp_decay_b' in self.train_cfg["runner"] else 1

        # fine_tune=True 表示微调基础策略网络，否则固定该网络参数，即不参与更新
        self.fine_tune = self.train_cfg["runner"]["fine_tune"] if 'fine_tune' in self.train_cfg["runner"] else None
        if self.fine_tune or self.fine_tune is None:
            self.optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)  # 5e-6
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.student.parameters()), lr=self.lr)

        self.num_base_obs = self.env_cfg["env"]["num_base_obs"]
        self.num_history = self.env_cfg["env"]["num_history"]
        self.obs_history = torch.zeros((self.env.num_envs, self.num_history * self.num_base_obs), dtype=torch.float,
                                       device=self.device)
        # print("Observation history shape: ", self.obs_history.shape)

        # 初始化日志 Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        if skip_reset is False:
            _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 新建文件夹 create folders
        dir_names = ['tensorboard', 'setting']
        dir_dict = create_folders(self.log_dir, dir_names)

        # 初始化 writer（用于 TensorBoard 日志记录）
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=dir_dict['tensorboard'], flush_secs=10)

        # 如果需要在随机的 episode 长度上初始化环境
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # 保存 cfg 到指定文件夹
        with open(os.path.join(dir_dict['setting'], 'env_config.json'), 'w') as f:
            f.write(json.dumps(self.env_cfg, sort_keys=False, indent=4, separators=(',', ': ')))
        with open(os.path.join(dir_dict['setting'], 'train_config.json'), 'w') as f:
            f.write(json.dumps(self.train_cfg, sort_keys=False, indent=4, separators=(',', ': ')))

        # 获取环境的观测
        # obs.shape: (num_envs, num_base_obs + num_terrain_height + num_extrinsics + ...) -> 包含外部环境信息的所有观测
        # obs_history.shape: (num_envs, num_base_obs * history_length) -> 历史本体感受观测
        # obs_history dim 1: concatenate(o_{t-H}, o_{t-H+1},..., o_{t-1}, o_{t})
        obs = self.env.get_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.num_base_obs:], obs[:, :self.num_base_obs]), dim=-1)
        assert self.obs_history.shape[1] == self.num_history * self.num_base_obs

        # 复制基础策略网络 MLP 参数
        self._copy_base_policy_params()

        obs, self.obs_history = obs.to(self.device), self.obs_history.to(self.device)
        self.student.train()  # switch to train mode
        self.teacher.eval()  # switch to eval mode

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # stundent 执行动作并与环境进行交互产生数据
        learning_rate = self.lr
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):  # 使用 load() 时，self.current_learning_iteration 可能非零
            start = time.time()
            # 数据采集 Rollout
            mean_action_loss = 0.
            mean_latent_loss = 0.
            for i in range(self.num_steps_per_env):
                # 分别获取 teacher 和 student 的 action 和 latent vector
                teacher_action, teacher_latent = self.teacher.act_inference(obs.detach())
                student_action, student_latent = self.student.act_inference(self.obs_history)

                obs, privileged_obs, rewards, dones, infos = self.env.step(student_action.detach())
                # 使用teacher的action进行训练，student网络不与环境进行交互
                # obs, privileged_obs, rewards, dones, infos = self.env.step(teacher_action.detach())
                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                    self.device), dones.to(self.device)

                # 监督学习: 计算 MSE loss
                action_loss = torch.mean((teacher_action - student_action) ** 2)
                latent_loss = torch.mean((teacher_latent - student_latent) ** 2)

                mean_action_loss += action_loss.item()
                mean_latent_loss += latent_loss.item()

                # 如果不是 fine-tune 则不计算 action_loss
                action_loss_scale = 1 if self.fine_tune or self.fine_tune is None else 0
                loss = action_loss * action_loss_scale + latent_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 环境 terminate 则重置 obs_history
                env_id = dones.nonzero(as_tuple=False).flatten()
                self.obs_history[env_id, :] = 0.
                self.obs_history = torch.cat((self.obs_history[:, self.num_base_obs:], obs[:, :self.num_base_obs]),
                                             dim=-1)

                # 更新 learning rate
                decay_factor = math.pow(self.exp_decay_a, (it / self.exp_decay_b))
                learning_rate = self.lr * decay_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate

                # 处理日志信息
                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    # 记录当前累积奖励和累积步数，并将 dones=1 对应索引的环境的 cur_reward_sum, cur_episode_length 清零
                    cur_reward_sum += rewards.detach()
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            mean_action_loss /= self.num_steps_per_env
            mean_latent_loss /= self.num_steps_per_env
            stop = time.time()
            iteration_time = stop - start

            if self.log_dir is not None:
                #  log 方法可以访问并打印 learn 方法中的所有局部变量，其中 locals() 用于获取 learn 方法中的局部变量的字典
                self.log(locals())
            if it % self.save_interval == 0:
                # 每隔一定步数保存模型
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))  # 保存模型

    def log(self, locs, width=80, pad=35):
        """ 记录和输出训练过程中的日志信息，包括各种损失值、性能指标、训练速度等

        :param locs: locals() -> 获取 learn 方法中的局部变量的字典
        :param width: 80
        :param pad: 35
        :return: None
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['iteration_time']
        iteration_time = locs['iteration_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['iteration_time']))

        self.writer.add_scalar('Loss/action_loss', locs['mean_action_loss'], locs['it'])
        self.writer.add_scalar('Loss/latent_loss', locs['mean_latent_loss'], locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/iteration time', locs['iteration_time'], locs['it'])
        self.writer.add_scalar('Perf/learning rate', locs['learning_rate'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection & learning: {locs['iteration_time']:.3f}s)\n"""
                          f"""{'Action loss:':>{pad}} {locs['mean_action_loss']:.4f}\n"""
                          f"""{'Latent loss:':>{pad}} {locs['mean_latent_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection & learning: {iteration_time:.3f}s)\n"""
                          f"""{'Action loss:':>{pad}} {locs['mean_action_loss']:.4f}\n"""
                          f"""{'Latent loss:':>{pad}} {locs['mean_latent_loss']:.4f}\n"""
                          )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        """ 周期性地调用 self.save 方法，在训练过程中保存模型的状态字典、优化器状态字典、当前迭代次数等信息 """
        torch.save({
            'model_state_dict': self.student.state_dict(),  # 当前训练过程中策略和值函数网络的权重
            'optimizer_state_dict': self.optimizer.state_dict(),  # 当前优化器的状态，包括学习率等信息
            'iter': self.current_learning_iteration,
            'infos': infos,  # 额外的信息，例如训练过程中的统计数据等
        }, path)

    def load(self, path, load_optimizer=True):
        """ 加载保存的模型，可选择是否同时加载优化器状态 """
        loaded_dict = torch.load(path)
        self.student.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """ 将模型切换到评估模式 eval() """
        self.student.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.student.to(device)
        return self.student.act_inference_student

    def _copy_base_policy_params(self):
        """ 复制基础策略的网络参数 """
        # 获取 teacher/student 网络中 actor 和 critic 部分的参数
        teacher_actor_state_dict = self.teacher.actor.state_dict()
        teacher_critic_state_dict = self.teacher.critic.state_dict()
        student_actor_params = self.student.actor.parameters()
        student_critic_params = self.student.critic.parameters()
        # print(self.student.actor.state_dict()['0.weight'], '\n')

        # 将教师网络的 actor/critic 参数复制到学生网络的相应部分
        for student_param, teacher_param in zip(student_actor_params, teacher_actor_state_dict.values()):
            student_param.data.copy_(teacher_param.data)
        for student_param, teacher_param in zip(student_critic_params, teacher_critic_state_dict.values()):
            student_param.data.copy_(teacher_param.data)
        # print(self.student.actor.state_dict()['0.weight'], '\n')

        # 微调学生网络
        if self.fine_tune:  # 设置 requires_grad 为 True，表示需要计算参数的梯度
            for param in self.student.parameters():
                param.requires_grad = True
            for param in self.student.critic.parameters():
                param.requires_grad = False
        else:
            # 解冻 encoder 参数
            for param in self.student.parameters():
                param.requires_grad = True
            # 固定 actor 的参数
            for param in self.student.actor.parameters():
                param.requires_grad = False
            for param in self.student.critic.parameters():
                param.requires_grad = False
