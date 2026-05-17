# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

from __future__ import annotations

import glob
import os
import sys

import toml


def _iter_local_isaac_asset_root_candidates() -> list[str]:
    """Return plausible local Isaac asset roots.

    Isaac Lab expects ``/persistent/isaac/asset_root/cloud`` to point to the directory that
    contains the local ``Isaac`` folder (for example ``~/4.5`` on this machine).
    """

    candidates: list[str] = []
    seen: set[str] = set()

    def _append(path: str | None):
        if not path:
            return
        resolved = os.path.abspath(os.path.expanduser(path))
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    for env_var in ("TACEX_ISAAC_ASSET_ROOT", "ISAAC_ASSET_ROOT", "ISAACLAB_ASSET_ROOT"):
        _append(os.environ.get(env_var))

    home_dir = os.path.expanduser("~")
    for version_dir in ("4.5", "4.5.0", "5.0", "5.0.0"):
        _append(os.path.join(home_dir, version_dir))

    panda_glob = os.path.join(home_dir, "*", "Isaac", "IsaacLab", "Robots", "FrankaEmika", "panda_instanceable.usd")
    for panda_usd_path in glob.glob(panda_glob):
        _append(os.path.join(os.path.dirname(panda_usd_path), "..", "..", "..", ".."))

    return candidates


def _find_local_isaac_asset_root() -> str | None:
    """Find a local Isaac asset root that contains the Franka Panda USD."""

    suffix = os.path.join("Isaac", "IsaacLab", "Robots", "FrankaEmika", "panda_instanceable.usd")
    for candidate in _iter_local_isaac_asset_root_candidates():
        if os.path.isfile(os.path.join(candidate, suffix)):
            return candidate
    return None


def _patch_loaded_isaaclab_asset_modules(asset_root: str):
    """Update already-imported Isaac Lab asset constants/configs to the local root."""

    isaac_dir = os.path.join(asset_root, "Isaac")
    isaaclab_dir = os.path.join(isaac_dir, "IsaacLab")

    assets_module = sys.modules.get("isaaclab.utils.assets")
    if assets_module is not None:
        assets_module.NUCLEUS_ASSET_ROOT_DIR = asset_root
        assets_module.NVIDIA_NUCLEUS_DIR = f"{asset_root}/NVIDIA"
        assets_module.ISAAC_NUCLEUS_DIR = isaac_dir
        assets_module.ISAACLAB_NUCLEUS_DIR = isaaclab_dir

    franka_module = sys.modules.get("isaaclab_assets.robots.franka")
    if franka_module is not None:
        franka_usd_path = os.path.join(isaaclab_dir, "Robots", "FrankaEmika", "panda_instanceable.usd")
        for cfg_name in ("FRANKA_PANDA_CFG", "FRANKA_PANDA_HIGH_PD_CFG"):
            cfg = getattr(franka_module, cfg_name, None)
            if cfg is not None and getattr(getattr(cfg, "spawn", None), "usd_path", None):
                cfg.spawn.usd_path = franka_usd_path


def _configure_local_isaac_asset_root():
    """Prefer a local Isaac asset tree over the default cloud HTTP root."""

    asset_root = _find_local_isaac_asset_root()
    if asset_root is None:
        return

    os.environ.setdefault("TACEX_ISAAC_ASSET_ROOT", asset_root)

    try:
        import carb
    except Exception:
        carb = None

    if carb is not None:
        carb.settings.get_settings().set("/persistent/isaac/asset_root/cloud", asset_root)

    _patch_loaded_isaaclab_asset_modules(asset_root)


_configure_local_isaac_asset_root()

# Conveniences to other module directories via relative paths
ISAACLAB_TASKS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

ISAACLAB_TASKS_METADATA = toml.load(os.path.join(ISAACLAB_TASKS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = ISAACLAB_TASKS_METADATA["package"]["version"]

##
# Register Gym environments.
##

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
