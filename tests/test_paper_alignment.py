"""Check the public training configurations against the paper architecture."""

import unittest

from legged_gym.envs.force_ts.SepEnc.task_config import ForceSepTSCfg, ForceSepTSCfgPPO
from legged_gym.envs.force_ts.OneEnc.task_config import ForceOneTSCfg
from legged_gym.envs.vanilla.vanilla_rl_ts.task_config import VanillaRLTSCfg


class PaperAlignmentTest(unittest.TestCase):
    def test_pa_loco_architecture_and_training_ranges(self):
        cfg, train = ForceSepTSCfg(), ForceSepTSCfgPPO()
        self.assertEqual((cfg.env.num_base_obs, cfg.env.num_history, cfg.env.num_disturbance_obs,
                          cfg.env.num_latent, cfg.env.num_residual_obs), (45, 50, 30, 22, 67))
        self.assertEqual((cfg.control.action_scale, cfg.control.res_action_scale), (0.25, 0.1))
        self.assertEqual((cfg.control.control_type, cfg.control.stiffness['joint'],
                          cfg.control.damping['joint']), ('P', 20, 0.5))
        self.assertEqual((cfg.domain_rand.added_mass_range, cfg.domain_rand.com_displacement_range_z,
                          cfg.domain_rand.max_push_ang_vel, cfg.domain_rand.push_force_noise),
                         ([-1., 1.], [-0.03, 0.03], 2.5, 2))
        self.assertEqual(cfg.commands.max_curriculum, 1.0)
        self.assertEqual((cfg.env.num_envs, train.runner.num_steps_per_env,
                          train.algorithm.num_mini_batches), (4096, 24, 6))
        self.assertEqual(train.policy.residual_hidden_dims, [256, 128, 64])

    def test_ablation_architecture(self):
        sefa = ForceOneTSCfg()
        robust = VanillaRLTSCfg()
        self.assertEqual((sefa.env.num_history, sefa.env.num_latent, sefa.env.num_extrinsic_obs),
                         (50, 22, 58))
        self.assertEqual((sefa.control.control_type, robust.control.control_type), ('P', 'P'))


if __name__ == '__main__':
    unittest.main()
