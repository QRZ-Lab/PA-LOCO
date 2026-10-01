"""Simulation rollout of trained PA-LOCO and ablation policies."""

import os

import isaacgym
import torch

from legged_gym.envs import task_registry
from legged_gym.utils import class_to_dict, get_args
from rsl_rl.modules import ActorCriticMlpEncoderResidual


def play(args):
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.mefa and args.task != "force_sep_t":
        raise ValueError("--mefa is available only for force_sep_t")
    if not args.student_policy or not os.path.isfile(args.student_policy):
        raise ValueError("--student_policy must point to an existing TorchScript file")
    if args.mefa or args.task in ("vanilla_rl_t", "force_one_t"):
        return play_student(args)
    if args.task != "force_sep_t":
        raise ValueError("Unsupported task")
    if not args.residual_checkpoint or not os.path.isfile(args.residual_checkpoint):
        raise ValueError("--residual_checkpoint must point to an existing checkpoint")
    env_cfg, train_cfg = task_registry.get_cfgs(name="force_sep_t")
    env_cfg.env.num_envs = args.num_envs or 16
    env_cfg.terrain.plane_pre_train = False
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.control.control_type = "P"
    env_cfg.control.is_res_training = True
    env_cfg.commands.ranges.lin_vel_x[1] = env_cfg.commands.max_curriculum
    env_cfg.domain_rand.max_push_force = env_cfg.domain_rand.max_push_force_curriculum
    env_cfg.control.policy_dir = args.student_policy
    env, env_cfg = task_registry.make_env(name="force_sep_t", args=args, env_cfg=env_cfg)

    cfg = class_to_dict(env_cfg)
    obs_cfg = cfg["env"]
    policy = ActorCriticMlpEncoderResidual(
        cfg, obs_cfg["num_actor_obs"], obs_cfg["num_privileged_obs"], obs_cfg["num_actions"],
        **class_to_dict(train_cfg.policy), **dict(class_to_dict(train_cfg.encoder), is_teacher=False)
    ).to(env.device)
    policy.load_state_dict(torch.load(args.residual_checkpoint, map_location=env.device)["model_state_dict"])
    policy.eval()

    observations, _ = env.reset()
    reward_sum = 0.0
    terminations = 0
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy.act_inference(observations)
            observations, _, rewards, dones, infos = env.step(actions)
            reward_sum += rewards.mean().item()
            terminations += (dones & ~infos["time_outs"]).sum().item()
    print(f"steps={args.steps} envs={env.num_envs} mean_step_reward={reward_sum / args.steps:.4f} "
          f"early_terminations={terminations}")


def play_student(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 16
    env_cfg.terrain.plane_pre_train = False
    env_cfg.control.control_type = "P"
    if args.task == "force_sep_t":
        env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
        env_cfg.domain_rand.max_push_force = env_cfg.domain_rand.max_push_force_curriculum
    if args.task != "vanilla_rl_t":
        env_cfg.commands.ranges.lin_vel_x[1] = env_cfg.commands.max_curriculum
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    policy = torch.jit.load(args.student_policy, map_location=env.device).eval()
    observations, _ = env.reset()
    width = env_cfg.env.num_base_obs
    history = torch.zeros(env.num_envs, width * env_cfg.env.num_history, device=env.device)
    history[:, -width:] = observations[:, :width]
    reward_sum = 0.0
    terminations = 0
    with torch.inference_mode():
        for _ in range(args.steps):
            output = policy(history)
            actions = output[0] if isinstance(output, tuple) else output
            observations, _, rewards, dones, infos = env.step(actions)
            reward_sum += rewards.mean().item()
            terminations += (dones & ~infos["time_outs"]).sum().item()
            history = torch.cat((history[:, width:], observations[:, :width]), dim=-1)
            reset = dones.bool()
            history[reset] = 0
            history[reset, -width:] = observations[reset, :width]
    print(f"steps={args.steps} envs={env.num_envs} mean_step_reward={reward_sum / args.steps:.4f} "
          f"early_terminations={terminations}")


if __name__ == "__main__":
    play(get_args())
