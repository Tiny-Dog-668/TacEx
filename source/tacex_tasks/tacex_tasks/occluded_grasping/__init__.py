"""
Occluded Grasping Environments:
Goal is to grasp and lift an object under partial visual occlusion.
"""

import importlib

import gymnasium as gym
from isaaclab.utils import configclass

from . import agents
from .vision_box import (
    OccludedGraspingVisionBlurBoxCfg,
    OccludedGraspingVisionDownsampleBoxCfg,
    OccludedGraspingVisionOnlyBoxCfg,
    OccludedGraspingVisionOnlyBoxEnv,
    OccludedGraspingVisionOnlyWristBoxCfg,
    OccludedGraspingVisionOnlyWristBoxEnv,
)
from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
    OccludedGraspingVisionFourTactileDownsampleBoxCfg,
    OccludedGraspingVisionFourTactileSelfOcclusionBoxCfg,
    OccludedGraspingVTAlphaBoxCfg,
    OccludedGraspingVTAlphaDownsampleBoxCfg,
    OccludedGraspingVTAlphaBoxEnv,
    OccludedGraspingVTConvexBoxCfg,
    OccludedGraspingVTConvexBoxEnv,
    OccludedGraspingVisionFourTactileWristBoxCfg,
    OccludedGraspingVisionFourTactileWristBoxEnv,
)
from .vt_tactile_box import (
    OccludedGraspingTactileProprioBoxCfg,
    OccludedGraspingTactileProprioBoxEnv,
)
from .vt_alpha_beta_box import (
    OccludedGraspingVTAlphaBetaBoxCfg,
    OccludedGraspingVTAlphaBetaBoxEnv,
)
from .vt_pair_box import (
    OccludedGraspingVTPairBoxCfg,
    OccludedGraspingVTPairBoxEnv,
)
from .vt_pair_gru_box import (
    OccludedGraspingVTPairGRUBoxCfg,
    OccludedGraspingVTPairGRUBoxEnv,
)
from .vt_cnn_recon_box import (
    OccludedGraspingVTCNNReconBoxCfg,
    OccludedGraspingVTCNNReconBoxEnv,
    OccludedGraspingVTTactileCrossAlphaReconBoxEnv,
    OccludedGraspingVTTactileCrossAlphaReconDownsampleBoxCfg,
)
from .vt_down_residual_box import (
    OccludedGraspingVTDownResidualBoxCfg,
    OccludedGraspingVTDownResidualBoxEnv,
)
from .vt_reliability_stage_box import (
    OccludedGraspingVTReliabilityStageBoxCfg,
    OccludedGraspingVTReliabilityStageBoxEnv,
)
from .vt_reliability_stage_strong_gate_box import (
    OccludedGraspingVTReliabilityStageStrongGateBoxCfg,
    OccludedGraspingVTReliabilityStageStrongGateBoxEnv,
)
from .vt_alpha_recon_box import (
    OccludedGraspingVTAlphaReconBoxCfg,
    OccludedGraspingVTAlphaReconBoxEnv,
)
from .vt_alpha_gru_box import (
    OccludedGraspingVTAlphaGRUBoxCfg,
    OccludedGraspingVTAlphaGRUBoxEnv,
    OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
    OccludedGraspingVTAlphaGRUDrawerOcclusionCubeCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionCubeCfg,
    OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg,
    _configure_cube_object,
    _configure_cuboid_object,
    _configure_soft_cube_object,
    _configure_soft_cuboid_object,
    _configure_soft_cylinder_object,
)
from .vt_gru_box import (
    OccludedGraspingVTGRUBoxCfg,
    OccludedGraspingVTGRUDownsampleBoxCfg,
    OccludedGraspingVTGRUBoxEnv,
)
from .vt_gru_extra_tactile_box import (
    OccludedGraspingVTGRUExtraTactileBoxCfg,
    OccludedGraspingVTGRUExtraTactileBoxEnv,
)
from .vt_gru_sparsh_down_depth_box import (
    OccludedGraspingVTGRUSparshDownDepthBoxCfg,
    OccludedGraspingVTGRUSparshDownDepthBoxEnv,
)
from .vt_alpha_gru_beta_box import (
    OccludedGraspingVTAlphaGRUBetaBoxCfg,
    OccludedGraspingVTAlphaGRUBetaBoxEnv,
)
from .vt_convex_gru_box import (
    OccludedGraspingVTConvexGRUBoxCfg,
    OccludedGraspingVTConvexGRUBoxEnv,
)
from .vt_convex_gru_beta_box import (
    OccludedGraspingVTConvexGRUBetaBoxCfg,
    OccludedGraspingVTConvexGRUBetaBoxEnv,
)
from .vt_alpha_dual_gru_box import (
    OccludedGraspingVTAlphaDualGRUBoxCfg,
    OccludedGraspingVTAlphaDualGRUBoxEnv,
)
from .vt_alpha_gru_visual_reliability_box import (
    OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionBoxCfg,
    OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionCubeCfg,
    OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg,
    OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionCubeCfg,
    OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxEnv,
)
from .vt_dual_alpha_recon_box import (
    OccludedGraspingVTDualAlphaReconBoxCfg,
    OccludedGraspingVTDualAlphaReconBoxEnv,
)
from .vt_gate_box import (
    OccludedGraspingVTGateAlphaBoxCfg,
    OccludedGraspingVTGateAlphaBoxEnv,
)
from .vt_hard_gate_box import (
    OccludedGraspingVTHardGateBoxCfg,
    OccludedGraspingVTHardGateBoxEnv,
)
from .vt_dual_cross_alpha_aux_box import (
    OccludedGraspingVTDualCrossAlphaAuxBoxEnv,
    OccludedGraspingVTDualCrossAlphaAuxDownsampleBoxCfg,
)
from .vt_tactile_cross_alpha_visual_tokens_box import (
    OccludedGraspingVTTactileCrossAlphaVisualTokensBoxEnv,
    OccludedGraspingVTTactileCrossAlphaVisualTokensDownsampleBoxCfg,
)
from .vt_tactile_cross_alpha_visual_predict_visible_box import (
    OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleBoxEnv,
    OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg,
)
from .vt_tactile_cross_alpha_aux_box import (
    OccludedGraspingVTTactileCrossAlphaAuxBoxEnv,
    OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg,
)
from .vt_tactile_cross_alpha_aux_gru_box import (
    OccludedGraspingVTTactileCrossAlphaAuxGRUBoxEnv,
    OccludedGraspingVTTactileCrossAlphaAuxGRUDownsampleBoxCfg,
)
from .vt_gelfusion_box import (
    OccludedGraspingVTGelFusionBoxCfg,
    OccludedGraspingVTGelFusionBoxEnv,
    OccludedGraspingVTGelFusionDownsampleBoxCfg,
)
from .vt_sparsh import (
    OccludedGraspingVisionFourTactileSparshBoxCfg,
    OccludedGraspingVisionFourTactileSparshBoxEnv,
)
from .vt_policy_token_transformer_box import (
    OccludedGraspingVTPolicyTokenTransformerBoxCfg,
    OccludedGraspingVTPolicyTokenTransformerBoxEnv,
)


def _patch_skrl_runner_for_custom_models() -> None:
    """Allow skrl Runner to instantiate custom models via module:Class strings."""
    try:
        from skrl.utils.runner.torch import Runner as TorchRunner
    except Exception:
        return

    if getattr(TorchRunner, "_tacex_custom_model_patch", False):
        return

    original_component = TorchRunner._component

    def _component(self, name: str):
        if isinstance(name, str) and ":" in name:
            module_path, class_name = name.split(":", 1)
                                                                                                                                                                                                                                                                                                                                                                                                                                
            def _instantiator(*args, **kwargs):
                return_source = kwargs.pop("return_source", False)
                if return_source:
                    return f"CustomModel({module_path}:{class_name})"
                cls = getattr(importlib.import_module(module_path), class_name)
                return cls(*args, **kwargs)

            return _instantiator
        return original_component(self, name)

    TorchRunner._component = _component                                                                                                                                                     
    TorchRunner._tacex_custom_model_patch = True


_patch_skrl_runner_for_custom_models()


def _register_task(task_id: str, entry_point: str, env_cfg_entry_point, skrl_cfg_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "skrl_cfg_entry_point": f"{agents.__name__}:{skrl_cfg_name}",
        },
    )


def _make_scene_object_cfg(base_cfg_cls, fusion_name: str, scene_name: str, object_name: str):
    """Create a config class for one scene/object variant without changing env logic."""
    include_drawer = scene_name == "Drawer-Occlusion"
    include_outer = scene_name == "Drawer-Occlusion"
    use_cube = object_name == "Cube"
    use_cuboid = object_name == "Cuboid"
    use_soft_cylinder = object_name == "SoftCylinder"
    use_soft_cube = object_name == "SoftCube"
    use_soft_cuboid = object_name == "SoftCuboid"
    class_name = f"OccludedGrasping{fusion_name.replace('-', '')}{scene_name.replace('-', '')}{object_name}Cfg"

    def __post_init__(self):
        base_post_init = getattr(base_cfg_cls, "__post_init__", None)
        if base_post_init is not None:
            base_post_init(self)
        if use_cube:
            _configure_cube_object(self)
        if use_cuboid:
            _configure_cuboid_object(self)
        if use_soft_cylinder:
            _configure_soft_cylinder_object(self)
        if use_soft_cube:
            _configure_soft_cube_object(self)
        if use_soft_cuboid:
            _configure_soft_cuboid_object(self)
        post_configure = getattr(self, "_post_configure_scene_object", None)
        if callable(post_configure):
            post_configure()

    attrs = {
        "__module__": __name__,
        "__post_init__": __post_init__,
        "include_drawer_walls": include_drawer,
        "include_outer_cabinet_panels": include_outer,
    }
    if use_cube:
        attrs.update(
            {
                "cube_side": 0.05,
                "can_radius": 0.025,
                "can_reset_root_z": 0.0548,
                "lift_reward_start_height": 0.0548,
                "success_height": 0.1048,
                "drop_after_success_penalty_weight": 150.0,
            }
        )
    if use_cuboid:
        attrs.update(
            {
                "cuboid_size": (0.05, 0.05, 0.06),
                "can_radius": 0.025,
                "can_reset_root_z": 0.0598,
                "lift_reward_start_height": 0.0598,
                "success_height": 0.1198,
                "drop_after_success_penalty_weight": 150.0,
            }
        )
    if use_soft_cube:
        attrs.update(
            {
                "cube_side": 0.05,
                "can_radius": 0.025,
                "can_reset_root_z": 0.0548,
                "lift_reward_start_height": 0.0548,
                "success_height": 0.1048,
                "drop_after_success_penalty_weight": 150.0,
            }
        )
    if use_soft_cuboid:
        attrs.update(
            {
                "cuboid_size": (0.05, 0.05, 0.06),
                "can_radius": 0.025,
                "can_reset_root_z": 0.0598,
                "lift_reward_start_height": 0.0598,
                "success_height": 0.1198,
                "drop_after_success_penalty_weight": 150.0,
            }
        )
    cfg_cls = configclass(type(class_name, (base_cfg_cls,), attrs))
    globals()[class_name] = cfg_cls
    return cfg_cls


def _register_scene_object_fusion_matrix() -> None:
    """Register task IDs as TacEx-{Fusion}-{Scene}-{Object}, with old IDs kept as aliases."""
    alpha_gru_entry = f"{__name__}.vt_alpha_gru_box:OccludedGraspingVTAlphaGRUBoxEnv"
    alpha_gru_beta_entry = f"{__name__}.vt_alpha_gru_beta_box:OccludedGraspingVTAlphaGRUBetaBoxEnv"
    visual_reliability_entry = (
        f"{__name__}.vt_alpha_gru_visual_reliability_box:"
        "OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxEnv"
    )
    fusion_specs = {
        "V": (
            f"{__name__}.vision_box:OccludedGraspingVisionOnlyBoxEnv",
            OccludedGraspingVisionOnlyBoxCfg,
            "ppo_v.yaml",
        ),
        "V-Downsample": (
            f"{__name__}.vision_box:OccludedGraspingVisionOnlyBoxEnv",
            OccludedGraspingVisionDownsampleBoxCfg,
            "ppo_v.yaml",
        ),
        "V-Blur": (
            f"{__name__}.vision_box:OccludedGraspingVisionOnlyBoxEnv",
            OccludedGraspingVisionBlurBoxCfg,
            "ppo_v.yaml",
        ),
        "V-Wrist": (
            f"{__name__}.vision_box:OccludedGraspingVisionOnlyWristBoxEnv",
            OccludedGraspingVisionOnlyWristBoxCfg,
            "ppo_v_wrist.yaml",
        ),
        "T": (
            f"{__name__}.vt_tactile_box:OccludedGraspingTactileProprioBoxEnv",
            OccludedGraspingTactileProprioBoxCfg,
            "ppo_tactile.yaml",
        ),
        "VT": (
            f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
            OccludedGraspingVisionFourTactileBoxCfg,
            "ppo_vt.yaml",
        ),
        "VT-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
            OccludedGraspingVisionFourTactileDownsampleBoxCfg,
            "ppo_vt_downsample.yaml",
        ),
        "VT-Router-MAE-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
            OccludedGraspingVisionFourTactileDownsampleBoxCfg,
            "ppo_vt_router_mae_downsample.yaml",
        ),
        "GelFusion": (
            f"{__name__}.vt_gelfusion_box:OccludedGraspingVTGelFusionBoxEnv",
            OccludedGraspingVTGelFusionBoxCfg,
            "ppo_vt_gelfusion.yaml",
        ),
        "GelFusion-Downsample": (
            f"{__name__}.vt_gelfusion_box:OccludedGraspingVTGelFusionBoxEnv",
            OccludedGraspingVTGelFusionDownsampleBoxCfg,
            "ppo_vt_gelfusion.yaml",
        ),
        "VT-Wrist": (
            f"{__name__}.vt_box:OccludedGraspingVisionFourTactileWristBoxEnv",
            OccludedGraspingVisionFourTactileWristBoxCfg,
            "ppo_vt_wrist.yaml",
        ),
        "VT-Pair": (
            f"{__name__}.vt_pair_box:OccludedGraspingVTPairBoxEnv",
            OccludedGraspingVTPairBoxCfg,
            "ppo_vt_pair.yaml",
        ),
        "VT-Pair-GRU": (
            f"{__name__}.vt_pair_gru_box:OccludedGraspingVTPairGRUBoxEnv",
            OccludedGraspingVTPairGRUBoxCfg,
            "ppo_vt_pair_gru.yaml",
        ),
        "CNN-Recon": (
            f"{__name__}.vt_cnn_recon_box:OccludedGraspingVTCNNReconBoxEnv",
            OccludedGraspingVTCNNReconBoxCfg,
            "ppo_vt_cnn_recon.yaml",
        ),
        "VT-Down-Residual": (
            f"{__name__}.vt_down_residual_box:OccludedGraspingVTDownResidualBoxEnv",
            OccludedGraspingVTDownResidualBoxCfg,
            "ppo_vt_down_residual.yaml",
        ),
        "Reliability-Stage": (
            f"{__name__}.vt_reliability_stage_box:OccludedGraspingVTReliabilityStageBoxEnv",
            OccludedGraspingVTReliabilityStageBoxCfg,
            "ppo_vt_reliability_stage.yaml",
        ),
        "Reliability-Stage-Strong-Gate": (
            f"{__name__}.vt_reliability_stage_strong_gate_box:OccludedGraspingVTReliabilityStageStrongGateBoxEnv",
            OccludedGraspingVTReliabilityStageStrongGateBoxCfg,
            "ppo_vt_reliability_stage_strong_gate.yaml",
        ),
        "Alpha": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaBoxCfg,
            "ppo_vt_alpha.yaml",
        ),
        "Visual-Cross-Alpha": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaBoxCfg,
            "ppo_vt_visual_cross_alpha.yaml",
        ),
        "Visual-Cross-Alpha-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_visual_cross_alpha.yaml",
        ),
        "Visual-Cross-Alpha-Tactile-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_visual_cross_alpha_tactile.yaml",
        ),
        "Tactile-Cross-Alpha": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaBoxCfg,
            "ppo_vt_tactile_cross_alpha.yaml",
        ),
        "Tactile-Cross-Alpha-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha.yaml",
        ),
        "Tactile-Cross-Alpha-Recon-Downsample": (
            f"{__name__}.vt_cnn_recon_box:OccludedGraspingVTTactileCrossAlphaReconBoxEnv",
            OccludedGraspingVTTactileCrossAlphaReconDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha_recon.yaml",
        ),
        "Tactile-Cross-Alpha-Aux-Downsample": (
            f"{__name__}.vt_tactile_cross_alpha_aux_box:OccludedGraspingVTTactileCrossAlphaAuxBoxEnv",
            OccludedGraspingVTTactileCrossAlphaAuxDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha_aux.yaml",
        ),
        "Tactile-Cross-Alpha-Aux-GRU-Downsample": (
            f"{__name__}.vt_tactile_cross_alpha_aux_gru_box:OccludedGraspingVTTactileCrossAlphaAuxGRUBoxEnv",
            OccludedGraspingVTTactileCrossAlphaAuxGRUDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha_aux_gru.yaml",
        ),
        "Tactile-Cross-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
            OccludedGraspingVisionFourTactileDownsampleBoxCfg,
            "ppo_vt_tactile_cross.yaml",
        ),
        "Tactile-Cross-Alpha-Visual-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha_visual.yaml",
        ),
        "Tactile-Cross-Alpha-Visual-PredictVisible-Downsample": (
            f"{__name__}.vt_tactile_cross_alpha_visual_predict_visible_box:OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleBoxEnv",
            OccludedGraspingVTTactileCrossAlphaVisualPredictVisibleDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha_visual_predict_visible.yaml",
        ),
        "Tactile-Cross-Alpha-VisualTokens-Downsample": (
            f"{__name__}.vt_tactile_cross_alpha_visual_tokens_box:OccludedGraspingVTTactileCrossAlphaVisualTokensBoxEnv",
            OccludedGraspingVTTactileCrossAlphaVisualTokensDownsampleBoxCfg,
            "ppo_vt_tactile_cross_alpha_visual_tokens.yaml",
        ),
        "Dual-Cross-Alpha": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaBoxCfg,
            "ppo_vt_dual_cross_alpha.yaml",
        ),
        "Dual-Cross-Alpha-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_dual_cross_alpha.yaml",
        ),
        "Dual-Cross-Alpha-Aux-Downsample": (
            f"{__name__}.vt_dual_cross_alpha_aux_box:OccludedGraspingVTDualCrossAlphaAuxBoxEnv",
            OccludedGraspingVTDualCrossAlphaAuxDownsampleBoxCfg,
            "ppo_vt_dual_cross_alpha_aux.yaml",
        ),
        "Dual-Cross-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_dual_cross.yaml",
        ),
        "Token-Self-Attn-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_token_self_attention.yaml",
        ),
        "Alpha-Downsample": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaDownsampleBoxCfg,
            "ppo_vt_alpha_downsample.yaml",
        ),
        "Alpha-Learnable-Tactile": (
            f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
            OccludedGraspingVTAlphaBoxCfg,
            "ppo_vt_alpha_learnable_tactile.yaml",
        ),
        "Alpha-Beta": (
            f"{__name__}.vt_alpha_beta_box:OccludedGraspingVTAlphaBetaBoxEnv",
            OccludedGraspingVTAlphaBetaBoxCfg,
            "ppo_vt_alpha_beta.yaml",
        ),
        "Alpha-GRU": (
            f"{__name__}.vt_alpha_gru_box:OccludedGraspingVTAlphaGRUBoxEnv",
            OccludedGraspingVTAlphaGRUBoxCfg,
            "ppo_vt_alpha_gru.yaml",
        ),
        "Alpha-GRU-Beta": (
            alpha_gru_beta_entry,
            OccludedGraspingVTAlphaGRUBetaBoxCfg,
            "ppo_vt_alpha_gru_beta.yaml",
        ),
        "Alpha-Dual-GRU": (
            f"{__name__}.vt_alpha_dual_gru_box:OccludedGraspingVTAlphaDualGRUBoxEnv",
            OccludedGraspingVTAlphaDualGRUBoxCfg,
            "ppo_vt_alpha_dual_gru.yaml",
        ),
        "Alpha-GRU-VisualReliability": (
            visual_reliability_entry,
            OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg,
            "ppo_vt_alpha_gru_visual_reliability.yaml",
        ),
        "Alpha-Recon": (
            f"{__name__}.vt_alpha_recon_box:OccludedGraspingVTAlphaReconBoxEnv",
            OccludedGraspingVTAlphaReconBoxCfg,
            "ppo_vt_alpha_recon.yaml",
        ),
        "Dual-Alpha-Recon": (
            f"{__name__}.vt_dual_alpha_recon_box:OccludedGraspingVTDualAlphaReconBoxEnv",
            OccludedGraspingVTDualAlphaReconBoxCfg,
            "ppo_vt_dual_alpha_recon.yaml",
        ),
        "Gate-Alpha": (
            f"{__name__}.vt_gate_box:OccludedGraspingVTGateAlphaBoxEnv",
            OccludedGraspingVTGateAlphaBoxCfg,
            "ppo_vt_gate_alpha.yaml",
        ),
        "Hard-Gate": (
            f"{__name__}.vt_hard_gate_box:OccludedGraspingVTHardGateBoxEnv",
            OccludedGraspingVTHardGateBoxCfg,
            "ppo_vt.yaml",
        ),
        "GRU": (
            f"{__name__}.vt_gru_box:OccludedGraspingVTGRUBoxEnv",
            OccludedGraspingVTGRUBoxCfg,
            "ppo_vt_gru.yaml",
        ),
        "GRU-Downsample": (
            f"{__name__}.vt_gru_box:OccludedGraspingVTGRUBoxEnv",
            OccludedGraspingVTGRUDownsampleBoxCfg,
            "ppo_vt_gru_downsample.yaml",
        ),
        "GRU-Extra-Tactile": (
            f"{__name__}.vt_gru_extra_tactile_box:OccludedGraspingVTGRUExtraTactileBoxEnv",
            OccludedGraspingVTGRUExtraTactileBoxCfg,
            "ppo_vt_gru_extra_tactile.yaml",
        ),
        "GRU-Sparsh-DownDepth": (
            f"{__name__}.vt_gru_sparsh_down_depth_box:OccludedGraspingVTGRUSparshDownDepthBoxEnv",
            OccludedGraspingVTGRUSparshDownDepthBoxCfg,
            "ppo_vt_gru_sparsh_down_depth.yaml",
        ),
        "Convex": (
            f"{__name__}.vt_box:OccludedGraspingVTConvexBoxEnv",
            OccludedGraspingVTConvexBoxCfg,
            "ppo_vt_convex.yaml",
        ),
        "Convex-GRU": (
            f"{__name__}.vt_convex_gru_box:OccludedGraspingVTConvexGRUBoxEnv",
            OccludedGraspingVTConvexGRUBoxCfg,
            "ppo_vt_convex_gru.yaml",
        ),
        "Convex-GRU-Beta": (
            f"{__name__}.vt_convex_gru_beta_box:OccludedGraspingVTConvexGRUBetaBoxEnv",
            OccludedGraspingVTConvexGRUBetaBoxCfg,
            "ppo_vt_convex_gru_beta.yaml",
        ),
        "Sparsh": (
            f"{__name__}.vt_sparsh:OccludedGraspingVisionFourTactileSparshBoxEnv",
            OccludedGraspingVisionFourTactileSparshBoxCfg,
            "ppo_vt_sparsh.yaml",
        ),
        "Policy-Token-Transformer": (
            f"{__name__}.vt_policy_token_transformer_box:OccludedGraspingVTPolicyTokenTransformerBoxEnv",
            OccludedGraspingVTPolicyTokenTransformerBoxCfg,
            "ppo_vt_policy_token_transformer.yaml",
        ),
    }

    explicit_cfgs = {
        ("Alpha-GRU", "Drawer-Occlusion", "Cylinder"): OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
        ("Alpha-GRU", "Drawer-Occlusion", "Cube"): OccludedGraspingVTAlphaGRUDrawerOcclusionCubeCfg,
        ("Alpha-GRU", "Self-Occlusion", "Cylinder"): OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg,
        ("Alpha-GRU", "Self-Occlusion", "Cube"): OccludedGraspingVTAlphaGRUSelfOcclusionCubeCfg,
        ("Alpha-GRU-VisualReliability", "Drawer-Occlusion", "Cylinder"): (
            OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionBoxCfg
        ),
        ("Alpha-GRU-VisualReliability", "Drawer-Occlusion", "Cube"): (
            OccludedGraspingVTAlphaGRUVisualReliabilityDrawerOcclusionCubeCfg
        ),
        ("Alpha-GRU-VisualReliability", "Self-Occlusion", "Cylinder"): (
            OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg
        ),
        ("Alpha-GRU-VisualReliability", "Self-Occlusion", "Cube"): (
            OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionCubeCfg
        ),
    }

    for fusion_name, (entry_point, base_cfg_cls, skrl_cfg) in fusion_specs.items():
        for scene_name in ("Drawer-Occlusion", "Self-Occlusion"):
            for object_name in ("Cylinder", "Cube", "Cuboid", "SoftCylinder", "SoftCube", "SoftCuboid"):
                cfg_cls = explicit_cfgs.get(
                    (fusion_name, scene_name, object_name),
                    _make_scene_object_cfg(base_cfg_cls, fusion_name, scene_name, object_name),
                )
                _register_task(
                    task_id=f"TacEx-{fusion_name}-{scene_name}-{object_name}",
                    entry_point=entry_point,
                    env_cfg_entry_point=cfg_cls,
                    skrl_cfg_name=skrl_cfg,
                )
                _register_task(
                    task_id=f"TacEx-VT-{fusion_name}-{scene_name}-{object_name}-v0",
                    entry_point=entry_point,
                    env_cfg_entry_point=cfg_cls,
                    skrl_cfg_name=skrl_cfg,
                )

    aliases = {
        "TacEx-VT-Alpha-GRU-Box-v0": (
            alpha_gru_entry,
            OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
            "ppo_vt_alpha_gru.yaml",
        ),
        "TacEx-VT-Alpha-GRU-Beta-Box-v0": (
            alpha_gru_beta_entry,
            OccludedGraspingVTAlphaGRUBetaBoxCfg,
            "ppo_vt_alpha_gru_beta.yaml",
        ),
        "TacEx-VT-Alpha-GRU-Drawer-Occlusion-Box-v0": (
            alpha_gru_entry,
            OccludedGraspingVTAlphaGRUDrawerOcclusionBoxCfg,
            "ppo_vt_alpha_gru.yaml",
        ),
        "TacEx-VT-Alpha-GRU-Self-Occlusion-Box-v0": (
            alpha_gru_entry,
            OccludedGraspingVTAlphaGRUSelfOcclusionBoxCfg,
            "ppo_vt_alpha_gru.yaml",
        ),
        "TacEx-VT-Alpha-GRU-VisualReliability-Self-Occlusion-Box-v0": (
            visual_reliability_entry,
            OccludedGraspingVTAlphaGRUVisualReliabilitySelfOcclusionBoxCfg,
            "ppo_vt_alpha_gru_visual_reliability.yaml",
        ),
    }
    for task_id, (entry_point, cfg_cls, skrl_cfg) in aliases.items():
        _register_task(task_id, entry_point, cfg_cls, skrl_cfg)


# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Box-v0 --num_envs 4 --enable_cameras env.tactile_encoder_type=resnet
# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Box-v0 --num_envs 4 --enable_cameras env.tactile_encoder_type=cnn
gym.register(
    id="TacEx-VT-Box-v0",
    entry_point=f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionFourTactileBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Pair-Box-v0 --num_envs 4 --enable_cameras
# VT-Pair:
#   down tactile pair: 2 * 256 = 512 -> 128
#   inner tactile pair: 2 * 256 = 512 -> 128
#   fused = concat(vision_256, down_128, inner_128, proprio_18)
gym.register(
    id="TacEx-VT-Pair-Box-v0",
    entry_point=f"{__name__}.vt_pair_box:OccludedGraspingVTPairBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTPairBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_pair.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Pair-GRU-Box-v0 --num_envs 4 --enable_cameras
# VT-Pair-GRU:
#   each tactile sensor uses a 10-step GRU encoder
#   down tactile pair: 2 * 128 = 256 -> 128
#   inner tactile pair: 2 * 128 = 256 -> 128
#   fused = concat(vision_256, down_128, inner_128, proprio_18)
gym.register(
    id="TacEx-VT-Pair-GRU-Box-v0",
    entry_point=f"{__name__}.vt_pair_gru_box:OccludedGraspingVTPairGRUBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTPairGRUBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_pair_gru.yaml",
    },
)

#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
gym.register(
    id="TacEx-VT-Self-Occlusion-Box-v0",
    entry_point=f"{__name__}.vt_box:OccludedGraspingVisionFourTactileBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionFourTactileSelfOcclusionBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Reliability-Stage-Box-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-VT-Reliability-Stage-Box-v0",
    entry_point=f"{__name__}.vt_reliability_stage_box:OccludedGraspingVTReliabilityStageBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTReliabilityStageBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_reliability_stage.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Reliability-Stage-Strong-Gate-Box-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-VT-Reliability-Stage-Strong-Gate-Box-v0",
    entry_point=f"{__name__}.vt_reliability_stage_strong_gate_box:OccludedGraspingVTReliabilityStageStrongGateBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTReliabilityStageStrongGateBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_reliability_stage_strong_gate.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Alpha-Box-v0 --num_envs 4 --enable_cameras
# VT-Alpha: concat-style gated fusion
#   alpha = sigmoid(MLP(proprio))
#   fused = concat((1 - alpha) * v_hat, alpha * t_hat, proprio)
gym.register(
    id="TacEx-VT-Alpha-Box-v0",
    entry_point=f"{__name__}.vt_box:OccludedGraspingVTAlphaBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTAlphaBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_alpha.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Alpha-Beta-Box-v0 --num_envs 4 --enable_cameras
# VT-Alpha-Beta:
#   alpha,beta = softmax(MLP(proprio))[down, inner]
#   fused = concat((1 - alpha - beta) * v_256, alpha * t_down_128, beta * t_inner_128, proprio)
gym.register(
    id="TacEx-VT-Alpha-Beta-Box-v0",
    entry_point=f"{__name__}.vt_alpha_beta_box:OccludedGraspingVTAlphaBetaBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTAlphaBetaBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_alpha_beta.yaml",
    },
)

# Composable naming for occluded grasping variants:
#   TacEx-VT-{Fusion}-{Scene}-{Object}-v0
# Scenes: Self-Occlusion, Drawer-Occlusion
# Objects: Cylinder, Cube, Cuboid
_register_scene_object_fusion_matrix()

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-GRU-Box-v0 --num_envs 4 --enable_cameras
# VT-GRU:
#   tactile sensors use a 10-step GRU encoder
#   fused = concat(v_256, t_gru_256, proprio)
gym.register(
    id="TacEx-VT-GRU-Box-v0",
    entry_point=f"{__name__}.vt_gru_box:OccludedGraspingVTGRUBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTGRUBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_gru.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Alpha-Dual-GRU-Box-v0 --num_envs 4 --enable_cameras
# VT-Alpha-Dual-GRU:
#   1. Keep tactile 10-step GRU temporal encoding
#   2. Add a 5-step GRU over third-person visual features
#   3. Keep the same alpha-gated fusion layout
gym.register(
    id="TacEx-VT-Alpha-Dual-GRU-Box-v0",
    entry_point=f"{__name__}.vt_alpha_dual_gru_box:OccludedGraspingVTAlphaDualGRUBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTAlphaDualGRUBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_alpha_dual_gru.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Alpha-Recon-Box-v0 --num_envs 4 --enable_cameras
# VT-Alpha-Recon:
#   1. Supervise task heads with GT: occlusion, down contact, object-side (left/center/right), inner contact
#   2. Build alpha from [task-head predictions, proprio]
#   3. fused = concat((1 - alpha) * v_hat, alpha * t_hat, proprio)
gym.register(
    id="TacEx-VT-Alpha-Recon-Box-v0",
    entry_point=f"{__name__}.vt_alpha_recon_box:OccludedGraspingVTAlphaReconBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTAlphaReconBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_alpha_recon.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Dual-Alpha-Recon-Box-v0 --num_envs 4 --enable_cameras
# VT-Dual-Alpha-Recon:
#   1. Two separate alpha gates for down/inner tactile branches
#   2. alpha_down <- [visual-head, down-head, proprio], alpha_inner <- [visual-head, inner-head, proprio]
#   3. fused = concat(vision_256, alpha_down * down_128, alpha_inner * inner_128)
gym.register(
    id="TacEx-VT-Dual-Alpha-Recon-Box-v0",
    entry_point=f"{__name__}.vt_dual_alpha_recon_box:OccludedGraspingVTDualAlphaReconBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTDualAlphaReconBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_dual_alpha_recon.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Gate-Alpha-Box-v0 --num_envs 4 --enable_cameras
# VT-Gate-Alpha:
#   1. Per-sensor tactile baseline / validity detection
#   2. alpha = tactile_valid * sigmoid(MLP(v_hat, t_hat, proprio, tactile_valid))
#   3. fused = concat((1 - alpha) * v_hat, alpha * t_hat, proprio)
gym.register(
    id="TacEx-VT-Gate-Alpha-Box-v0",
    entry_point=f"{__name__}.vt_gate_box:OccludedGraspingVTGateAlphaBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTGateAlphaBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_gate_alpha.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Hard-Gate-Box-v0 --num_envs 4 --enable_cameras
# VT-Hard-Gate:
#   1. Per-sensor depth baseline / contact detection
#   2. Each tactile feature is zeroed when its contact ratio is below threshold
gym.register(
    id="TacEx-VT-Hard-Gate-Box-v0",
    entry_point=f"{__name__}.vt_hard_gate_box:OccludedGraspingVTHardGateBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTHardGateBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Convex-Box-v0 --num_envs 4 --enable_cameras
# VT-Convex: convex-combination fusion
#   alpha = sigmoid(MLP(proprio))
#   fused = concat((1 - alpha) * v_hat + alpha * t_hat, proprio)
gym.register(
    id="TacEx-VT-Convex-Box-v0",
    entry_point=f"{__name__}.vt_box:OccludedGraspingVTConvexBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTConvexBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_convex.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Convex-GRU-Box-v0 --num_envs 4 --enable_cameras
# VT-Convex-GRU:
#   tactile sensors use a 10-step GRU encoder
#   fused = concat((1 - alpha) * v_512 + alpha * t_gru_512, proprio)
gym.register(
    id="TacEx-VT-Convex-GRU-Box-v0",
    entry_point=f"{__name__}.vt_convex_gru_box:OccludedGraspingVTConvexGRUBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTConvexGRUBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_convex_gru.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Convex-GRU-Beta-Box-v0 --num_envs 4 --enable_cameras
# VT-Convex-GRU-Beta:
#   tactile sensors use a 10-step GRU encoder
#   fused = (1 - alpha - beta) * v_512 + alpha * t_down_512 + beta * t_inner_512
gym.register(
    id="TacEx-VT-Convex-GRU-Beta-Box-v0",
    entry_point=f"{__name__}.vt_convex_gru_beta_box:OccludedGraspingVTConvexGRUBetaBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVTConvexGRUBetaBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_convex_gru_beta.yaml",
    },
)

# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Sparsh-Box-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-VT-Sparsh-Box-v0",
    entry_point=f"{__name__}.vt_sparsh:OccludedGraspingVisionFourTactileSparshBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionFourTactileSparshBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_sparsh.yaml",
    },
)
#env.sparsh_encoder_name=dino_vitbase  dino_vitsmall  dinov2_vitbase ijepa_vitsmall ijepa_vitbase


# python scripts/reinforcement_learning/skrl/train.py --task TacEx-VT-Wrist-Box-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-VT-Wrist-Box-v0",
    entry_point=f"{__name__}.vt_box:OccludedGraspingVisionFourTactileWristBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionFourTactileWristBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_vt_wrist.yaml",
    }, 
)


# python scripts/reinforcement_learning/skrl/train.py --task TacEx-V-Box-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-V-Box-v0",
    entry_point=f"{__name__}.vision_box:OccludedGraspingVisionOnlyBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionOnlyBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_v.yaml",
    },
)


# python scripts/reinforcement_learning/skrl/train.py --task TacEx-V-Wrist-Box-v0 --num_envs 4 --enable_cameras
gym.register(
    id="TacEx-V-Wrist-Box-v0",
    entry_point=f"{__name__}.vision_box:OccludedGraspingVisionOnlyWristBoxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": OccludedGraspingVisionOnlyWristBoxCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:ppo_v_wrist.yaml",
    },
)
