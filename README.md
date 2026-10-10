# PA-LOCO (IROS 2024)

_Training code repository_ for [**PA-LOCO: Learning Perturbation-Adaptive Locomotion for Quadruped Robots**](https://qrz-lab.github.io/PA-LOCO/) 

_Main authors_: **Zhiyuan Xiao**, **Xinyu Zhang**, **Xiang Zhou**, and **Qingrui Zhang**.

_PDF download link_:· [IEEE paper](https://ieeexplore.ieee.org/document/10801753/) · [arXiv preprint](https://arxiv.org/abs/2407.04224)

## Methods

| Paper method | Task | Training |
| --- | --- | --- |
| PA-LOCO | `force_sep_t` | Teacher, student, residual |
| MEFA (separate encoders, no residual) | `force_sep_t` | Teacher, student |
| SEFA (single encoder) | `force_one_t` | Teacher, student |
| Robust (domain randomization) | `vanilla_rl_t` | Teacher, student |

The PA-LOCO teacher encodes terrain heights, robot dynamics, and a 10-step external-force history separately. The student adapts from 50 steps of proprioceptive history. The residual policy takes the 45-dimensional observation and 22-dimensional student latent as input. Task names follow the original training workspace.

## Setup

Linux, an NVIDIA GPU, Python 3.8, CUDA-capable PyTorch, and Isaac Gym Preview 4 are required. Install Isaac Gym separately, then run:

```bash
python -m pip install -r requirements.txt
export ISAAC_GYM_ROOT=/path/to/isaacgym
export PYTHONPATH="$ISAAC_GYM_ROOT/python:$PWD"
python -m unittest discover -s tests -v
```

## Train PA-LOCO

Run these stages in order. Training outputs are written under `logs/`.

```bash
python -m legged_gym.scripts.rma.train_teacher --task=force_sep_t --headless
python -m legged_gym.scripts.rma.train_student --task=force_sep_t --load_run=<TEACHER_RUN> --headless
python -m legged_gym.scripts.rma.train_residual --task=force_sep_t --headless \
  --teacher_checkpoint=logs/force_sep_t/<TEACHER_RUN>/model_6000.pt \
  --student_checkpoint=logs/force_sep_ts/<STUDENT_RUN>/model_6000.pt \
  --student_policy=logs/force_sep_ts/<STUDENT_RUN>/policy/policy.pt
```

Use the checkpoint names produced by your runs. The student stage exports `policy/policy.pt`. For SEFA or Robust, use the corresponding task name with `train_teacher` and `train_student`. The `force_sep_t` student alone implements MEFA.

A quick execution check can use `--num_envs=2 --max_iterations=1` for each stage; this checks the training path, not policy performance.

## Evaluate in simulation

```bash
python -m legged_gym.scripts.rma.play --task=force_sep_t --headless \
  --student_policy=logs/force_sep_ts/<STUDENT_RUN>/policy/policy.pt \
  --residual_checkpoint=logs/force_sep_tsr/<RESIDUAL_RUN>/model_6000.pt
```

For MEFA, SEFA, or Robust, pass `--mefa --task=force_sep_t`, `--task=force_one_t`, or `--task=vanilla_rl_t` with the corresponding `--student_policy`. Use `--steps` and `--num_envs` to set rollout length and environment count.

The Go1 URDF, meshes, PPO implementation, and notices are included. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
