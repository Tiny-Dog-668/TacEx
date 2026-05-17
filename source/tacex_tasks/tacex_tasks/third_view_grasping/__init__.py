"""Third-view grasping environments."""

import gymnasium as gym

from . import agents
from .privileged_box import (
    OccludedGraspingPrivilegedBoxCfg,
    OccludedGraspingPrivilegedBoxEnv,
)
from .privileged_wrist_box import (
    OccludedGraspingPrivilegedWristBoxCfg,
    OccludedGraspingPrivilegedWristBoxEnv,
)
from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)
from .wrist_only_box import (
    OccludedGraspingWristOnlyBoxCfg,
    OccludedGraspingWristOnlyBoxEnv,
)
from .third_view_proprio_box import (
    OccludedGraspingThirdViewProprioBoxCfg,
    OccludedGraspingThirdViewProprioBoxEnv,
)
from .vt_concat_box import (
    OccludedGraspingVTConcatBoxCfg,
    OccludedGraspingVTConcatBoxEnv,
)
from .third_view_tactile_concat_box import (
    OccludedGraspingThirdViewTactileConcatBoxCfg,
    OccludedGraspingThirdViewTactileConcatBoxEnv,
)
from .third_view_tactile_concat_gate_box import (
    OccludedGraspingThirdViewTactileConcatGateBoxCfg,
    OccludedGraspingThirdViewTactileConcatGateBoxEnv,
)
from .third_view_tactile_temporal_gate_box import (
    OccludedGraspingThirdViewTactileTemporalGateBoxCfg,
    OccludedGraspingThirdViewTactileTemporalGateBoxEnv,
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-v0 --num_envs 4 --enable_cameras
# Reward settings follow occluded_grasping/vt_box.py.
gym.register(
    id="TacEx-Third-View-Grasping-v0",
    entry_point=f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionFourTactileBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Privileged-v0 --num_envs 4
gym.register(
    id="TacEx-Third-View-Grasping-Privileged-v0",
    entry_point=f"{__name__}.privileged_box:OccludedGraspingPrivilegedBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingPrivilegedBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_privileged.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Privileged-Wrist-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Third-View-Grasping-Privileged-Wrist-v0",
    entry_point=f"{__name__}.privileged_wrist_box:OccludedGraspingPrivilegedWristBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingPrivilegedWristBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_privileged_wrist.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Wrist-Only-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Third-View-Grasping-Wrist-Only-v0",
    entry_point=f"{__name__}.wrist_only_box:OccludedGraspingWristOnlyBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingWristOnlyBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_wrist_only.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Proprio-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Third-View-Grasping-Proprio-v0",
    entry_point=f"{__name__}.third_view_proprio_box:OccludedGraspingThirdViewProprioBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingThirdViewProprioBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_third_proprio.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-VT-Concat-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Third-View-Grasping-VT-Concat-v0",
    entry_point=f"{__name__}.vt_concat_box:OccludedGraspingVTConcatBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTConcatBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_concat.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Tactile-Concat-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Third-View-Grasping-Tactile-Concat-v0",
    entry_point=f"{__name__}.third_view_tactile_concat_box:OccludedGraspingThirdViewTactileConcatBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingThirdViewTactileConcatBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_third_tactile_concat.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Tactile-Concat-Gate-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-Third-View-Grasping-Tactile-Concat-Gate-v0",
    entry_point=f"{__name__}.third_view_tactile_concat_gate_box:OccludedGraspingThirdViewTactileConcatGateBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingThirdViewTactileConcatGateBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_third_tactile_concat_gate.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-Third-View-Grasping-Tactile-Temporal-Gate-v0 --num_envs 4 --enable_cameras --headless
gym.register(
    id="TacEx-Third-View-Grasping-Tactile-Temporal-Gate-v0",
    entry_point=f"{__name__}.third_view_tactile_temporal_gate_box:OccludedGraspingThirdViewTactileTemporalGateBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingThirdViewTactileTemporalGateBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_third_tactile_temporal_gate.yaml",
    },
)
