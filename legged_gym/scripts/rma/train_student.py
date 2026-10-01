import os

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.force_ts.SepEnc.task_config import ForceSepTSCfg
from legged_gym.envs.force_ts.OneEnc.task_config import ForceOneTSCfg
from legged_gym.envs.vanilla.vanilla_rl_ts.task_config import VanillaRLTSCfg

from legged_gym.envs import *
from legged_gym.utils import get_args, class_to_dict, task_registry, export_policy_as_jit


def process_cfg(env_cfg):
    env_cfg.env.num_envs = 2048
    if isinstance(env_cfg, (ForceSepTSCfg, ForceOneTSCfg)):
        env_cfg.terrain.plane_pre_train = False
    # env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    paper_student_cfg = isinstance(env_cfg, (ForceSepTSCfg, VanillaRLTSCfg, ForceOneTSCfg))
    env_cfg.control.control_type = 'P' if paper_student_cfg else 'actuator_net'
    if env_cfg.commands.curriculum:
        if not isinstance(env_cfg, (ForceSepTSCfg, ForceOneTSCfg)):
            env_cfg.commands.ranges.lin_vel_x[0] = -env_cfg.commands.max_curriculum
            env_cfg.commands.ranges.lin_vel_y[0] = -env_cfg.commands.max_curriculum
            env_cfg.commands.ranges.lin_vel_y[1] = env_cfg.commands.max_curriculum
        env_cfg.commands.ranges.lin_vel_x[1] = env_cfg.commands.max_curriculum
    return env_cfg


def train(args):
    # 通过 get_cfgs() 获取与任务相关的 cfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = process_cfg(env_cfg)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # teacher runner
    train_cfg.runner.resume = True
    teacher_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args,
                                                              train_cfg=train_cfg)
    teacher = teacher_runner.alg.actor_critic.to(env.device)  # teacher 网络

    # 构造用于训练 student network 的 student runner
    train_cfg.runner.resume = False
    student_runner, train_cfg = task_registry.make_student_runner(env=env,  env_cfg=env_cfg, args=args,
                                                                  train_cfg=train_cfg, teacher=teacher,
                                                                  save_interval=200)

    # 调用student网络中的learn函数进行策略学习
    student_runner.learn(num_learning_iterations=args.max_iterations or train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    export_policy_as_jit(student_runner.student, os.path.join(student_runner.log_dir, 'policy'))


if __name__ == '__main__':
    args = get_args()
    train(args)
