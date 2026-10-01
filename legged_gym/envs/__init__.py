"""Tasks used by the PA-LOCO paper and its comparison policies."""

from legged_gym.envs.vanilla.vanilla_rl_ts.task import VanillaRLTS
from legged_gym.envs.vanilla.vanilla_rl_ts.task_config import VanillaRLTSCfg, VanillaRLTSCfgPPO
from legged_gym.envs.force_ts.OneEnc.task import ForceTS as ForceOneTS
from legged_gym.envs.force_ts.OneEnc.task_config import ForceOneTSCfg, ForceOneTSCfgPPO
from legged_gym.envs.force_ts.SepEnc.task import ForceTS as ForceSepTS
from legged_gym.envs.force_ts.SepEnc.task_config import ForceSepTSCfg, ForceSepTSCfgPPO
from legged_gym.utils.task_registry import task_registry

task_registry.register("vanilla_rl_t", VanillaRLTS, VanillaRLTSCfg(), VanillaRLTSCfgPPO())
task_registry.register("force_one_t", ForceOneTS, ForceOneTSCfg(), ForceOneTSCfgPPO())
task_registry.register("force_sep_t", ForceSepTS, ForceSepTSCfg(), ForceSepTSCfgPPO())
