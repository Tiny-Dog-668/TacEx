from __future__ import annotations

import os
import re
import shlex
from datetime import datetime
from typing import Any, Iterable


def _get_value(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    sentinel = object()
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, sentinel)
        else:
            current = getattr(current, part, sentinel)
        if current is sentinel:
            return default
    return current


def _normalize_shape(spec: Any) -> tuple[int, ...] | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        return (int(spec),)
    if isinstance(spec, (list, tuple)):
        try:
            return tuple(int(value) for value in spec)
        except Exception:
            return None
    shape = getattr(spec, "shape", None)
    if shape is not None:
        try:
            return tuple(int(value) for value in shape)
        except Exception:
            return None
    return None


def _shape_flat_dim(shape: tuple[int, ...] | None) -> int | None:
    if shape is None:
        return None
    total = 1
    for value in shape:
        total *= int(value)
    return total


def _shape_text(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "-"
    if len(shape) == 1:
        return str(shape[0])
    return "x".join(str(value) for value in shape)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_format_value(val)}" for key, val in value.items()) + "}"
    return str(value)


def _append_section(lines: list[str], title: str, items: Iterable[tuple[str, Any]]) -> None:
    section_items = [(label, value) for label, value in items if value not in (None, "", [], {}, ()) and value != "-"]
    if not section_items:
        return
    lines.append(f"[{title}]")
    for label, value in section_items:
        lines.append(f"{label}: {_format_value(value)}")
    lines.append("")


def _obs_specs(env_cfg: Any) -> dict[str, tuple[int, ...] | None]:
    observation_space = _get_value(env_cfg, "observation_space")
    if hasattr(observation_space, "spaces"):
        return {str(key): _normalize_shape(space) for key, space in observation_space.spaces.items()}
    if isinstance(observation_space, dict):
        return {str(key): _normalize_shape(value) for key, value in observation_space.items()}
    return {}


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _extract_state_keys(expression: Any) -> list[str]:
    if not isinstance(expression, str):
        return []
    return re.findall(r"STATES\[['\"]([^'\"]+)['\"]\]", expression)


def _policy_cfg(agent_cfg: dict) -> dict:
    return agent_cfg.get("models", {}).get("policy", {}) or {}


def _policy_feature_keys(env_cfg: Any, agent_cfg: dict) -> list[str]:
    obs_specs = _obs_specs(env_cfg)
    policy_cfg = _policy_cfg(agent_cfg)
    keys: list[str] = []

    for single_key in ("vision_key", "proprio_key"):
        value = policy_cfg.get(single_key)
        if isinstance(value, str) and value in obs_specs:
            keys.append(value)

    for multi_key in ("vision_keys", "tactile_keys", "vector_keys"):
        value = policy_cfg.get(multi_key, [])
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item in obs_specs:
                    keys.append(item)

    network = policy_cfg.get("network", [])
    if isinstance(network, list):
        for node in network:
            if not isinstance(node, dict):
                continue
            node_name = node.get("name")
            if isinstance(node_name, str) and node_name in obs_specs and node_name != "net":
                keys.append(node_name)
            keys.extend(key for key in _extract_state_keys(node.get("input")) if key in obs_specs)

    if not keys:
        keys = [key for key in obs_specs if not key.startswith("critic_")]

    return _ordered_unique(key for key in keys if not key.startswith("critic_"))


def _group_feature_keys(keys: Iterable[str]) -> dict[str, list[str]]:
    groups = {"vision": [], "tactile": [], "vector": [], "other": []}
    for key in keys:
        lower = key.lower()
        if "tactile" in lower:
            groups["tactile"].append(key)
        elif "proprio" in lower or "vector" in lower:
            groups["vector"].append(key)
        elif any(token in lower for token in ("vision", "third", "wrist", "camera")):
            groups["vision"].append(key)
        else:
            groups["other"].append(key)
    return groups


def _feature_items(obs_specs: dict[str, tuple[int, ...] | None], keys: Iterable[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for key in keys:
        shape = obs_specs.get(key)
        flat_dim = _shape_flat_dim(shape)
        if shape is None:
            items.append((key, "shape=unknown"))
        elif len(shape) == 1:
            items.append((key, f"dim={shape[0]}"))
        else:
            items.append((key, f"shape={_shape_text(shape)}, flat_dim={flat_dim}"))
    return items


def _tactile_feature_source_map(env_cfg: Any, tactile_keys: Iterable[str]) -> dict[str, str]:
    """Best-effort description of what raw modality feeds each tactile feature key."""
    tactile_keys = list(tactile_keys)
    if not tactile_keys:
        return {}

    left_types = set(_get_value(env_cfg, "gsmini_left.data_types", []) or [])
    right_types = set(_get_value(env_cfg, "gsmini_right.data_types", []) or [])
    left_down_types = set(_get_value(env_cfg, "gsmini_left_down.data_types", []) or [])
    right_down_types = set(_get_value(env_cfg, "gsmini_right_down.data_types", []) or [])
    sparsh_name = str(_get_value(env_cfg, "sparsh_encoder_name", "") or "").strip()
    is_mixed_hybrid = (
        sparsh_name
        and "tactile_rgb" in left_types
        and "tactile_rgb" in right_types
        and left_down_types == {"camera_depth"}
        and right_down_types == {"camera_depth"}
    )

    tactile_encoder_type = str(_get_value(env_cfg, "tactile_encoder_type", "") or "").strip().lower()
    encoder_label = {
        "resnet": "shared tactile ResNet18",
        "cnn": "shared tactile CNN",
    }.get(tactile_encoder_type, "shared tactile encoder")

    result: dict[str, str] = {}
    for key in tactile_keys:
        lower = key.lower()
        if is_mixed_hybrid and "left_down" in lower:
            result[key] = "down tactile camera_depth -> shared 1ch depth CNN"
            continue
        if is_mixed_hybrid and "right_down" in lower:
            result[key] = "down tactile camera_depth -> shared 1ch depth CNN"
            continue
        if is_mixed_hybrid and "left" in lower:
            result[key] = f"inner tactile_rgb -> shared SparshFrozenEncoder ({sparsh_name})"
            continue
        if is_mixed_hybrid and "right" in lower:
            result[key] = f"inner tactile_rgb -> shared SparshFrozenEncoder ({sparsh_name})"
            continue
        if "left_down" in lower:
            source = "down tactile camera_depth"
        elif "right_down" in lower:
            source = "down tactile camera_depth"
        elif "left" in lower:
            source = "inner tactile_rgb"
        elif "right" in lower:
            source = "inner tactile_rgb"
        else:
            source = "tactile input"
        result[key] = f"{source} -> {encoder_label}"
    return result


def _total_flat_dim(obs_specs: dict[str, tuple[int, ...] | None], keys: Iterable[str]) -> int | None:
    dims = [_shape_flat_dim(obs_specs.get(key)) for key in keys]
    if not dims or any(dim is None for dim in dims):
        return None
    return int(sum(dim for dim in dims if dim is not None))


def _camera_resolution(env_cfg: Any) -> str | None:
    height = _get_value(env_cfg, "third_person_camera.height")
    width = _get_value(env_cfg, "third_person_camera.width")
    if height is None or width is None:
        return None
    return f"{height}x{width}"


def _visual_input_source(env_cfg: Any) -> str | None:
    prim_path = str(_get_value(env_cfg, "third_person_camera.prim_path", "") or "")
    if not prim_path:
        return None
    if "wrist_camera" in prim_path:
        return "wrist RGB camera"
    return "third-person RGB camera"


def _vision_encoder_summary(env_cfg: Any, obs_specs: dict[str, tuple[int, ...] | None], vision_keys: list[str]) -> str | None:
    if not vision_keys:
        return None
    if any(key.endswith("_tokens") for key in vision_keys):
        return "ResNet18 spatial backbone -> visual tokens"
    if any("resnet" in key.lower() for key in vision_keys):
        return "ResNet18 global backbone -> pooled visual feature"
    if any("vision_obs" in key.lower() for key in vision_keys):
        return "raw vision image -> policy-side CNN"
    return f"derived from observation keys {vision_keys}"


def _tactile_encoder_summary(env_cfg: Any, tactile_keys: list[str]) -> str | None:
    if not tactile_keys:
        return None
    sparsh_name = _get_value(env_cfg, "sparsh_encoder_name")
    left_types = set(_get_value(env_cfg, "gsmini_left.data_types", []) or [])
    right_types = set(_get_value(env_cfg, "gsmini_right.data_types", []) or [])
    left_down_types = set(_get_value(env_cfg, "gsmini_left_down.data_types", []) or [])
    right_down_types = set(_get_value(env_cfg, "gsmini_right_down.data_types", []) or [])
    is_mixed_hybrid = (
        sparsh_name
        and "tactile_rgb" in left_types
        and "tactile_rgb" in right_types
        and left_down_types == {"camera_depth"}
        and right_down_types == {"camera_depth"}
    )
    if is_mixed_hybrid:
        return f"inner tactile: shared SparshFrozenEncoder ({sparsh_name}); down tactile: shared 1ch depth CNN"
    if sparsh_name:
        return f"SparshFrozenEncoder ({sparsh_name})"
    tactile_encoder_type = _get_value(env_cfg, "tactile_encoder_type")
    if tactile_encoder_type == "resnet":
        return "shared tactile ResNet18 encoder"
    if tactile_encoder_type == "cnn":
        return "shared tactile CNN encoder"
    if any(key.endswith("_tokens") for key in tactile_keys):
        return "tokenized tactile encoder"
    if any("resnet" in key.lower() for key in tactile_keys):
        return "encoder emits resnet-style tactile features"
    return f"derived from observation keys {tactile_keys}"


def _fusion_summary(agent_cfg: dict) -> tuple[str | None, str | None]:
    policy_cfg = _policy_cfg(agent_cfg)
    policy_class = str(policy_cfg.get("class", "") or "")
    network = policy_cfg.get("network", [])

    if "vt_alpha_policy" in policy_class.lower() or "VTAlphaPolicy" in policy_class:
        return (
            "alpha-gated late fusion",
            "concat((1-alpha) * vision_proj, alpha * tactile_proj, proprio)",
        )
    if "vt_convex_policy" in policy_class.lower() or "VTConvexPolicy" in policy_class:
        return (
            "convex alpha fusion",
            "(1-alpha) * vision_proj + alpha * tactile_proj, then concat proprio",
        )
    if "vt_cross_policy" in policy_class.lower() or "CrossAttentionPolicy" in policy_class:
        return (
            "vision-tactile cross-attention",
            (
                f"vision_token_hw={policy_cfg.get('vision_token_hw')} "
                f"tactile_token_hw={policy_cfg.get('tactile_token_hw')} "
                f"embed_dim={policy_cfg.get('embed_dim')} depth={policy_cfg.get('depth')} heads={policy_cfg.get('heads')}"
            ),
        )

    if isinstance(network, list):
        last_net = None
        for node in network:
            if isinstance(node, dict) and node.get("name") == "net":
                last_net = node
        if isinstance(last_net, dict):
            input_expr = last_net.get("input")
            if isinstance(input_expr, str) and "concatenate(" in input_expr:
                return (
                    "feature concatenation + MLP",
                    f"input={input_expr}; layers={last_net.get('layers')}",
                )

    if policy_class:
        return ("custom policy", policy_class)
    return (None, None)


def write_training_summary(
    log_dir: str,
    env_cfg: Any,
    agent_cfg: dict,
    args_cli: Any,
    hydra_args: list[str],
    algorithm: str,
    original_argv: list[str] | None = None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)

    raw_command = ""
    if original_argv:
        raw_command = " ".join(shlex.quote(arg) for arg in original_argv)

    obs_specs = _obs_specs(env_cfg)
    policy_keys = _policy_feature_keys(env_cfg, agent_cfg)
    grouped_keys = _group_feature_keys(policy_keys)

    vision_total = _total_flat_dim(obs_specs, grouped_keys["vision"])
    tactile_total = _total_flat_dim(obs_specs, grouped_keys["tactile"])
    vector_total = _total_flat_dim(obs_specs, grouped_keys["vector"] + grouped_keys["other"])
    fusion_method, fusion_detail = _fusion_summary(agent_cfg)

    summary_path = os.path.join(log_dir, "training_summary.txt")
    lines: list[str] = [
        "Training Summary",
        "================",
        "",
    ]

    _append_section(
        lines,
        "Run",
        (
            ("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("task", getattr(args_cli, "task", None)),
            ("algorithm", algorithm),
            ("seed", _get_value(env_cfg, "seed")),
            ("sim_device", _get_value(env_cfg, "sim.device")),
            ("num_envs", _get_value(env_cfg, "scene.num_envs")),
            ("log_dir", log_dir),
            ("raw_command", raw_command),
            ("hydra_overrides", hydra_args),
        ),
    )

    _append_section(
        lines,
        "Encoders",
        (
            ("visual_input_source", _visual_input_source(env_cfg)),
            ("visual_input_resolution", _camera_resolution(env_cfg)),
            ("vision_encoder", _vision_encoder_summary(env_cfg, obs_specs, grouped_keys["vision"])),
            ("vision_total_flat_dim", vision_total),
            ("tactile_input_resolution", _shape_text(_normalize_shape(_get_value(env_cfg, "tactile_img_res_hw")))),
            ("tactile_encoder", _tactile_encoder_summary(env_cfg, grouped_keys["tactile"])),
            ("tactile_total_flat_dim", tactile_total),
            ("vector_total_flat_dim", vector_total),
        ),
    )

    _append_section(lines, "Vision Features", _feature_items(obs_specs, grouped_keys["vision"]))
    _append_section(lines, "Tactile Features", _feature_items(obs_specs, grouped_keys["tactile"]))
    _append_section(
        lines,
        "Tactile Sources",
        tuple(_tactile_feature_source_map(env_cfg, grouped_keys["tactile"]).items()),
    )
    _append_section(lines, "Vector Features", _feature_items(obs_specs, grouped_keys["vector"] + grouped_keys["other"]))

    _append_section(
        lines,
        "Fusion",
        (
            ("policy_class", _get_value(agent_cfg, "models.policy.class", "GaussianMixin")),
            ("policy_output", _get_value(agent_cfg, "models.policy.output")),
            ("policy_clip_actions", _get_value(agent_cfg, "models.policy.clip_actions")),
            ("fusion_method", fusion_method),
            ("fusion_detail", fusion_detail),
            ("fused_dim", _get_value(agent_cfg, "models.policy.fused_dim")),
            ("vision_key", _get_value(agent_cfg, "models.policy.vision_key")),
            ("vision_keys", _get_value(agent_cfg, "models.policy.vision_keys")),
            ("tactile_keys", _get_value(agent_cfg, "models.policy.tactile_keys")),
            ("vector_keys", _get_value(agent_cfg, "models.policy.vector_keys")),
            ("embed_dim", _get_value(agent_cfg, "models.policy.embed_dim")),
            ("depth", _get_value(agent_cfg, "models.policy.depth")),
            ("heads", _get_value(agent_cfg, "models.policy.heads")),
            ("gate_hidden_dim", _get_value(agent_cfg, "models.policy.gate_hidden_dim")),
        ),
    )

    _append_section(
        lines,
        "Training",
        (
            ("trainer_timesteps", _get_value(agent_cfg, "trainer.timesteps")),
            ("rollouts", _get_value(agent_cfg, "agent.rollouts")),
            ("learning_epochs", _get_value(agent_cfg, "agent.learning_epochs")),
            ("mini_batches", _get_value(agent_cfg, "agent.mini_batches")),
            ("batch_size", _get_value(agent_cfg, "agent.batch_size")),
            ("gradient_steps", _get_value(agent_cfg, "agent.gradient_steps")),
            ("learning_rate", _get_value(agent_cfg, "agent.learning_rate")),
            ("learning_rate_scheduler", _get_value(agent_cfg, "agent.learning_rate_scheduler")),
            ("experiment_name", _get_value(agent_cfg, "agent.experiment.experiment_name")),
        ),
    )

    _append_section(
        lines,
        "Rewards",
        (
            ("reach_weight", _get_value(env_cfg, "reach_weight")),
            ("lift_weight", _get_value(env_cfg, "lift_weight")),
            ("success_reward_weight", _get_value(env_cfg, "success_reward_weight")),
        ),
    )

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return summary_path
