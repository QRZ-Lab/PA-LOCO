"""Train the paper's third-stage residual policy from a teacher/student pair."""

import os

import isaacgym
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import class_to_dict, get_args
from rsl_rl.modules import ActorCriticMlpEncoder, ActorCriticSepMlpEncoder


def train(args):
    if args.task != "force_sep_t":
        raise ValueError("Residual training is available only for force_sep_t")
    for option in ("teacher_checkpoint", "student_checkpoint", "student_policy"):
        path = getattr(args, option)
        if not path or not os.path.isfile(path):
            raise ValueError(f"--{option} must point to an existing file")
    env_cfg, train_cfg = task_registry.get_cfgs(name="force_sep_t")
    env_cfg.terrain.plane_pre_train = False
    env_cfg.control.control_type = "P"
    env_cfg.control.is_res_training = True
    env_cfg.commands.ranges.lin_vel_x[1] = env_cfg.commands.max_curriculum
    env_cfg.control.policy_dir = args.student_policy
    env, env_cfg = task_registry.make_env(name="force_sep_t", args=args, env_cfg=env_cfg)

    cfg = class_to_dict(env_cfg)
    policy_cfg = class_to_dict(train_cfg.policy)
    encoder_cfg = class_to_dict(train_cfg.encoder)
    obs = cfg["env"]
    model_args = (cfg, obs["num_actor_obs"], obs["num_privileged_obs"], obs["num_actions"])
    teacher = ActorCriticSepMlpEncoder(*model_args, **policy_cfg, **encoder_cfg).to(env.device)
    teacher.load_state_dict(torch.load(args.teacher_checkpoint, map_location=env.device)["model_state_dict"])

    student = ActorCriticMlpEncoder(*model_args, **policy_cfg, **dict(encoder_cfg, is_teacher=False)).to(env.device)
    student.load_state_dict(torch.load(args.student_checkpoint, map_location=env.device)["model_state_dict"])
    student.critic.load_state_dict(teacher.critic.state_dict())

    train_cfg.runner.resume = False
    train_cfg.runner.save_interval = 500
    runner, train_cfg = task_registry.make_residual_runner(
        env=env, env_cfg=env_cfg, args=args, train_cfg=train_cfg, student=student
    )
    runner.learn(num_learning_iterations=args.max_iterations or train_cfg.runner.max_iterations,
                 init_at_random_ep_len=True)


if __name__ == "__main__":
    train(get_args())
