import time
import os
import json
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPOResidual
from rsl_rl.modules import ActorCriticMlpEncoder, ActorCriticSepMlpEncoder, ActorCriticRecurrentMlpEncoder, ActorCriticMlpEncoderResidual
from rsl_rl.env import VecEnv
from rsl_rl.utils import create_folders


class ResidualRunner:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 student,
                 env_cfg=None,
                 log_dir=None,
                 device='cpu'):

        # 初始化训练配置、算法配置、策略配置等
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.encoder_cfg = train_cfg["encoder"] if 'encoder' in train_cfg else {}
        self.device = device
        self.env = env
        self.env_cfg = {} if env_cfg is None else env_cfg
        self.train_cfg = train_cfg
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        # %%%%%% 根据配置文件中所指定的策略类名/算法类名/环境的观测和动作空间，动态选择并实例化对应的模型和算法 %%%%%%%%%%%%%%%%

        # 第二阶段的 student 网络
        self.student: ActorCriticMlpEncoder = student
        # 指定待训练的 student (residual)
        actor_critic_class = eval(train_cfg["runner"]["residual_policy_class_name"])
        self.student_res: ActorCriticMlpEncoderResidual = actor_critic_class(self.env_cfg,
                                                                             self.env_cfg["env"]["num_actor_obs"],
                                                                             self.env_cfg["env"]["num_privileged_obs"],
                                                                             self.env_cfg["env"]["num_actions"],
                                                                             **self.train_cfg["policy"],
                                                                             **self.train_cfg["encoder"]).to(env.device)

        # PPO 算法针对 student_res 进行优化. 重新指定优化器，仅针对 base policy 进行优化
        # self.alg 变量的预期类型是 PPO 类，'**self.alg_cfg' -> 使用字典中的键值对作为关键字参数
        alg_class = eval(self.cfg["residual_algorithm_class_name"])  # 'PPOResidual'
        self.alg: PPOResidual = alg_class(self.student_res, device=self.device, **self.alg_cfg)

        # 初始化一些训练过程中需要的参数
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_base_obs = self.env_cfg["env"]["num_base_obs"]
        self.num_history = self.env_cfg["env"]["num_history"]
        self.num_latent = self.env_cfg["env"]["num_latent"]
        self.num_actions = self.env_cfg["env"]["num_actions"]
        # self.obs_history = torch.zeros((self.env.num_envs, self.num_history * self.num_base_obs), dtype=torch.float, device=self.device)

        # 初始化存储 RolloutStorage 类. obs_rollout 存储历史关节信息
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env_cfg["env"]["num_residual_obs"]], [self.env.num_privileged_obs], [self.env.num_actions])

        # 初始化日志 Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

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
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # 保存 cfg 到指定文件夹
        with open(os.path.join(dir_dict['setting'], 'env_config.json'), 'w') as f:
            f.write(json.dumps(self.env_cfg, sort_keys=False, indent=4, separators=(',', ': ')))
        with open(os.path.join(dir_dict['setting'], 'train_config.json'), 'w') as f:
            f.write(json.dumps(self.train_cfg, sort_keys=False, indent=4, separators=(',', ': ')))

        # 获取环境的初始观测
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        # 选择待更新网络模块以及定义优化器
        self._copy_policy_params()
        self._process_student_res_params()  # 仅更新 residual 网络
        self.alg.optimizer = torch.optim.Adam(self.student_res.parameters(), lr=self.alg.learning_rate)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):  # 使用 load() 时，self.current_learning_iteration 可能非零
            start = time.time()
            # 数据采集 Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # %%%%%% 智能体与环境交互 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    # 1. alg.act():
                    #       - 输入为历史观测信息
                    #       - 每次交互智能体生成动作并将 actions, values, actions_log_prob, action_mean/sigma, obs 等
                    #         信息记录到 transistion 中
                    # 2. env.step():
                    #       - 强化学习环境根据 actions 返回 obs, rewards, dones, infos 等信息
                    #       - obs: ot, lt, at
                    #       - privileged_obs: ot, privileged info
                    actions = self.alg.act(obs, critic_obs)  # 生成动作并将信息记录到 transition 中
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # 处理日志信息
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        # 记录当前累积奖励和累积步数，并将 dones=1 对应索引的环境的 cur_reward_sum, cur_episode_length 清零
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start  # 数据采集时间

                # %%%%%% 开始学习阶段 Learning step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # 1. alg.compute_returns():
                #       - 根据 critic_obs 计算最新状态价值
                #       - 调用 self.storage.compute_returns() 计算广义优势估计 GAE
                start = stop
                self.alg.compute_returns(critic_obs)

            # %%%%%% 策略更新 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # 1. self.alg.update():
            #       - 从 self.storage 储存的 num_steps_per_envs 步长的数据池中生成小批量数据
            #       - 计算 KL 散度
            #       - 计算 loss: loss 由 surrogate_loss, value_loss, entropy_loss 组成
            #       - 梯度反向传播, 裁减梯度, 优化器更新参数
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start  # 策略学习时间
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
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

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
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                          # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                          # f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          # f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

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
            'model_state_dict': self.alg.actor_critic.state_dict(),  # 当前训练过程中策略和值函数网络的权重
            'optimizer_state_dict': self.alg.optimizer.state_dict(),  # 当前优化器的状态，包括学习率等信息
            'iter': self.current_learning_iteration,
            'infos': infos,  # 额外的信息，例如训练过程中的统计数据等
            }, path)

    def load(self, path, load_optimizer=True):
        """ 加载保存的模型，可选择是否同时加载优化器状态 """
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        """ 获取用于推理的模型 """
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def _copy_policy_params(self):
        # 获取 student 网络中的参数
        student_actor_state_dict = self.student.actor.state_dict()
        student_critic_state_dict = self.student.critic.state_dict()
        student_encoder_state_dict = self.student.encoder.state_dict()
        # print(student_actor_state_dict['0.bias'][:10])

        student_res_actor_params = self.student_res.actor.parameters()
        student_res_critic_params = self.student_res.critic.parameters()
        student_res_encoder_params = self.student_res.encoder.parameters()
        # print(self.student_res.actor.state_dict()['0.bias'][:10])

        # 复制网络参数
        for student_res_param, student_param in zip(student_res_actor_params, student_actor_state_dict.values()):
            student_res_param.data.copy_(student_param.data)
        for student_res_param, student_param in zip(student_res_critic_params, student_critic_state_dict.values()):
            student_res_param.data.copy_(student_param.data)
        for student_res_param, student_param in zip(student_res_encoder_params, student_encoder_state_dict.values()):
            student_res_param.data.copy_(student_param.data)
        # print(self.student_res.actor.state_dict()['0.bias'][:10])

    def _process_student_res_params(self):
        for param in self.student_res.encoder.parameters():
            param.requires_grad = False
        for param in self.student_res.actor.parameters():
            param.requires_grad = False
        # 解冻参数参与更新
        for param in self.student_res.residual.parameters():
            param.requires_grad = True
        for param in self.student_res.critic.parameters():
            param.requires_grad = True
