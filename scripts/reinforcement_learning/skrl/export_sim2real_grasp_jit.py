"""Export a TacEx sim2real grasp skrl actor as a TorchScript module.

By default the exported module is end-to-end for deployment:

- action_history: [N, A]
- proprio_obs: [N, P]
- wrist_rgb: [N, H, W, 3] uint8

It outputs the actor mean actions with shape [N, A], where A is the task action
dimension. Cube policies are required to apply tanh inside the policy model, so
their exported means are bounded to [-1, 1] before environment-side action
scaling, IK post-processing, and safety gating.
"""

import argparse
import json
import os

import gymnasium as gym
import skrl
import torch
import torch.nn.functional as F
from packaging import version

from vision_encoder_artifact import (
    ENCODER_ARTIFACT_FILENAME,
    ENCODER_MANIFEST_FILENAME,
    STRICT_SIM2REAL_CUBE_TASKS,
    checkpoint_run_dir,
    load_saved_agent_config,
    load_saved_env_config,
    load_verified_vision_encoder,
    module_state_dict_sha256,
    normalized_policy_output_expression,
    sha256_file,
    validate_live_env_against_policy_contract,
    validate_live_vision_contract,
    validate_sim2real_policy_contract,
)

from isaaclab.app import AppLauncher


SUPPORTED_TASKS = (
    "TacEx-Sim2Real-Grasp-v0",
    "TacEx-Sim2Real-Cube-Grasp-v0",
    "TacEx-Sim2Real-Cube-Real-Alignment-v0",
)
BOUNDED_CUBE_TASKS = set(STRICT_SIM2REAL_CUBE_TASKS)
TRACE_MAX_ABS_TOLERANCE = 1e-5


parser = argparse.ArgumentParser(description="Export a TacEx sim2real grasp skrl actor to TorchScript.")
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-Sim2Real-Grasp-v0",
    help=f"Task name. Supported tasks: {', '.join(SUPPORTED_TASKS)}.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the skrl checkpoint.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help=(
        "Output path for the exported TorchScript module. Defaults to "
        "<checkpoint_dir>/exported/policy_actor_e2e_<checkpoint_stem>.pt"
    ),
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to initialize.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for the skrl experiment.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for the skrl experiment.",
)
parser.add_argument(
    "--input_mode",
    type=str,
    default="rgb",
    choices=["rgb", "features"],
    help="Export an end-to-end RGB actor or a smaller actor that consumes precomputed wrist_resnet features.",
)
parser.add_argument(
    "--encoder_artifact",
    type=str,
    default=None,
    help=(
        "Frozen ResNet18 state-dict artifact. Defaults to "
        "<checkpoint_run>/params/vision_encoder_resnet18.pt."
    ),
)
parser.add_argument(
    "--encoder_manifest",
    type=str,
    default=None,
    help=(
        "Frozen ResNet18 manifest. Defaults to "
        "<checkpoint_run>/params/vision_encoder_resnet18.json."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The sim2real export task always instantiates a TiledCamera for the policy visual encoder.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    raise ValueError("This exporter currently supports only torch-based skrl checkpoints.")

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry
from skrl.utils.spaces.torch import compute_space_size

import tacex_tasks  # noqa: F401


SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. Install supported version using "
        f"'pip install skrl>={SKRL_VERSION}'"
    )


def _validate_policy_export_contract(task: str, experiment_cfg: dict) -> bool:
    """Validate that Cube checkpoint config contains the training-time tanh."""
    policy_cfg = experiment_cfg.get("models", {}).get("policy", {})
    bounded_mean = normalized_policy_output_expression(policy_cfg.get("output")) == "tanh(actions)"
    if task in BOUNDED_CUBE_TASKS and not bounded_mean:
        raise RuntimeError(
            "Cube policy checkpoint was not trained with models.policy.output='tanh(ACTIONS)'. "
            "Refusing to add tanh only during export. Retrain with the bounded Actor config."
        )
    return bounded_mean


def _check_finite_actions(actions: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(actions).all().item()):
        raise RuntimeError(f"{label} contains NaN or Inf.")


def _check_action_bounds(actions: torch.Tensor, label: str, tolerance: float = 1e-6) -> None:
    _check_finite_actions(actions, label)
    max_abs = float(torch.max(torch.abs(actions)).item()) if actions.numel() else 0.0
    if max_abs > 1.0 + tolerance:
        raise RuntimeError(f"{label} violates bounded Actor contract: max_abs={max_abs:.8f}")


class Sim2RealGraspFeatureActorWrapper(torch.nn.Module):
    """Wrapper exporting the actor that consumes precomputed wrist_resnet features."""

    def __init__(self, policy, state_preprocessor, observation_space):
        super().__init__()
        self.policy = policy
        self.state_preprocessor = state_preprocessor
        self._use_state_preprocessor = state_preprocessor is not None

        self._total_obs_dim = int(compute_space_size(observation_space, occupied_size=True))
        self._action_history_start, self._action_history_end = self._get_slice(observation_space, "action_history")
        self._proprio_obs_start, self._proprio_obs_end = self._get_slice(observation_space, "proprio_obs")
        self._wrist_resnet_start, self._wrist_resnet_end = self._get_slice(observation_space, "wrist_resnet")

        self.action_history_dim = self._action_history_end - self._action_history_start
        self.proprio_obs_dim = self._proprio_obs_end - self._proprio_obs_start
        self.wrist_resnet_dim = self._wrist_resnet_end - self._wrist_resnet_start

    @staticmethod
    def _get_slice(observation_space, target_key: str) -> tuple[int, int]:
        start = 0
        for key in sorted(observation_space.keys()):
            size = int(compute_space_size(observation_space[key], occupied_size=True))
            end = start + size
            if key == target_key:
                return start, end
            start = end
        raise KeyError(f"Observation key not found in observation_space: {target_key}")

    def forward(
        self,
        action_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        wrist_resnet: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = wrist_resnet.shape[0]
        device = wrist_resnet.device
        dtype = wrist_resnet.dtype

        states = torch.zeros((batch_size, self._total_obs_dim), device=device, dtype=dtype)
        states[:, self._action_history_start : self._action_history_end] = action_history
        states[:, self._proprio_obs_start : self._proprio_obs_end] = proprio_obs
        states[:, self._wrist_resnet_start : self._wrist_resnet_end] = wrist_resnet

        if self._use_state_preprocessor:
            states = self.state_preprocessor(states, train=False)

        mean_actions, _, _ = self.policy.compute({"states": states}, role="policy")
        return mean_actions


class Sim2RealGraspRgbActorWrapper(Sim2RealGraspFeatureActorWrapper):
    """Wrapper exporting an end-to-end RGB actor for sim2real deployment."""

    def __init__(
        self,
        policy,
        state_preprocessor,
        observation_space,
        vision_encoder,
        imgnet_mean: torch.Tensor,
        imgnet_std: torch.Tensor,
    ):
        super().__init__(policy, state_preprocessor, observation_space)
        self.vision_encoder = vision_encoder
        self.register_buffer("_imgnet_mean", imgnet_mean.view(1, 3, 1, 1).to(dtype=torch.float32))
        self.register_buffer("_imgnet_std", imgnet_std.view(1, 3, 1, 1).to(dtype=torch.float32))

    def forward(
        self,
        action_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        wrist_rgb: torch.Tensor,
    ) -> torch.Tensor:
        x = wrist_rgb.to(dtype=torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._imgnet_mean) / self._imgnet_std
        wrist_resnet = self.vision_encoder(x).view(x.shape[0], self.wrist_resnet_dim)
        return super().forward(action_history, proprio_obs, wrist_resnet)


def main():
    algorithm = args_cli.algorithm.lower()

    if args_cli.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task {args_cli.task!r}. Expected one of: {', '.join(SUPPORTED_TASKS)}"
        )

    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )

    run_dir = checkpoint_run_dir(resume_path)
    params_dir = run_dir / "params"
    manifest_path = params_dir / ENCODER_MANIFEST_FILENAME
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_contract = validate_sim2real_policy_contract(
        run_manifest,
        task=args_cli.task,
        params_dir=params_dir,
    )
    experiment_cfg, checkpoint_cfg_path = load_saved_agent_config(params_dir)
    print(f"[INFO] Loaded agent config saved with checkpoint: {checkpoint_cfg_path}")
    bounded_actor_mean = _validate_policy_export_contract(args_cli.task, experiment_cfg)
    if normalized_policy_output_expression(
        experiment_cfg.get("models", {}).get("policy", {}).get("output")
    ) != run_contract.get("policy_output_expression"):
        raise RuntimeError("Saved agent output expression does not match the run policy contract.")

    env_cfg, checkpoint_env_cfg_path = load_saved_env_config(params_dir)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    print(f"[INFO] Loaded environment config saved with checkpoint: {checkpoint_env_cfg_path}")

    output_path = args_cli.output
    if output_path is None:
        checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
        default_name = (
            f"policy_actor_e2e_{checkpoint_stem}.pt"
            if args_cli.input_mode == "rgb"
            else f"policy_actor_{checkpoint_stem}.pt"
        )
        output_path = os.path.join(os.path.dirname(resume_path), "exported", default_name)
    output_path = os.path.abspath(output_path)
    if not output_path.endswith(".pt"):
        raise ValueError(f"TorchScript output path must end with .pt: {output_path}")
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    protected_paths = {
        os.path.realpath(resume_path),
        os.path.realpath(params_dir / ENCODER_ARTIFACT_FILENAME),
        os.path.realpath(manifest_path),
    }
    if args_cli.encoder_artifact:
        protected_paths.add(os.path.realpath(args_cli.encoder_artifact))
    if args_cli.encoder_manifest:
        protected_paths.add(os.path.realpath(args_cli.encoder_manifest))
    for candidate_path in (output_path, metadata_path):
        if os.path.realpath(candidate_path) in protected_paths:
            raise ValueError(f"Export output must not overwrite a source artifact: {candidate_path}")
        if os.path.commonpath(
            (os.path.realpath(candidate_path), os.path.realpath(params_dir))
        ) == os.path.realpath(params_dir):
            raise ValueError(
                f"Export output must not be written inside the source params directory: {candidate_path}"
            )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    policy_cfg = experiment_cfg.get("models", {}).get("policy", {})
    if bool(policy_cfg.get("recurrent", False)) or algorithm == "ppo_rnn":
        raise NotImplementedError("This exporter currently supports only non-recurrent skrl policies.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "ppo_rnn"]:
            env = multi_agent_to_single_agent(env)

        base_env = env.unwrapped
        validate_live_env_against_policy_contract(base_env, run_contract)
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        runner = Runner(env, experiment_cfg)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.set_running_mode("eval")

        policy = runner.agent.policy
        state_preprocessor = getattr(runner.agent, "_state_preprocessor", None)

        policy.eval().cpu()
        policy.device = torch.device("cpu")
        if state_preprocessor is not None:
            state_preprocessor.eval().cpu()

        vision_encoder = getattr(base_env, "_resnet18", None)
        imgnet_mean = getattr(base_env, "_imgnet_mean", None)
        imgnet_std = getattr(base_env, "_imgnet_std", None)
        if vision_encoder is None or imgnet_mean is None or imgnet_std is None:
            raise RuntimeError("Sim2real export requires the environment to expose a ResNet18 vision encoder.")
        encoder_artifact_path = os.path.abspath(
            args_cli.encoder_artifact or os.path.join(params_dir, ENCODER_ARTIFACT_FILENAME)
        )
        encoder_manifest_path = os.path.abspath(
            args_cli.encoder_manifest or os.path.join(params_dir, ENCODER_MANIFEST_FILENAME)
        )
        encoder_manifest = load_verified_vision_encoder(
            vision_encoder,
            artifact_path=encoder_artifact_path,
            manifest_path=encoder_manifest_path,
        )
        validate_sim2real_policy_contract(
            encoder_manifest,
            task=args_cli.task,
            params_dir=params_dir,
        )
        if encoder_manifest.get("state_dict_sha256") != run_manifest.get("state_dict_sha256"):
            raise RuntimeError("Selected encoder artifact does not match the checkpoint run manifest.")
        validate_live_vision_contract(base_env, encoder_manifest)

        if args_cli.input_mode == "rgb":
            vision_encoder.eval().cpu()
            export_wrapper = Sim2RealGraspRgbActorWrapper(
                policy=policy,
                state_preprocessor=state_preprocessor,
                observation_space=env.observation_space,
                vision_encoder=vision_encoder,
                imgnet_mean=imgnet_mean.cpu(),
                imgnet_std=imgnet_std.cpu(),
            )
            rgb_height = int(base_env.cfg.wrist_camera.height)
            rgb_width = int(base_env.cfg.wrist_camera.width)
        else:
            encoder_manifest = dict(encoder_manifest)
            encoder_manifest["embedded"] = False
            export_wrapper = Sim2RealGraspFeatureActorWrapper(
                policy=policy,
                state_preprocessor=state_preprocessor,
                observation_space=env.observation_space,
            )
            rgb_height = None
            rgb_width = None

        export_wrapper.eval().cpu()

        example_action_history = torch.zeros((1, export_wrapper.action_history_dim), dtype=torch.float32)
        example_proprio_obs = torch.zeros((1, export_wrapper.proprio_obs_dim), dtype=torch.float32)

        with torch.inference_mode():
            probe_action_history = torch.randn((2, export_wrapper.action_history_dim), dtype=torch.float32)
            probe_proprio_obs = torch.randn((2, export_wrapper.proprio_obs_dim), dtype=torch.float32)

            if args_cli.input_mode == "rgb":
                example_wrist_rgb = torch.zeros((1, rgb_height, rgb_width, 3), dtype=torch.uint8)
                traced = torch.jit.trace(
                    export_wrapper,
                    (example_action_history, example_proprio_obs, example_wrist_rgb),
                    strict=False,
                )
                probe_wrist_rgb = torch.randint(0, 256, (2, rgb_height, rgb_width, 3), dtype=torch.uint8)
                ref_actions = export_wrapper(probe_action_history, probe_proprio_obs, probe_wrist_rgb)
                jit_actions = traced(probe_action_history, probe_proprio_obs, probe_wrist_rgb)
            else:
                example_wrist_resnet = torch.zeros((1, export_wrapper.wrist_resnet_dim), dtype=torch.float32)
                traced = torch.jit.trace(
                    export_wrapper,
                    (example_action_history, example_proprio_obs, example_wrist_resnet),
                    strict=False,
                )
                probe_wrist_resnet = torch.randn((2, export_wrapper.wrist_resnet_dim), dtype=torch.float32)
                ref_actions = export_wrapper(probe_action_history, probe_proprio_obs, probe_wrist_resnet)
                jit_actions = traced(probe_action_history, probe_proprio_obs, probe_wrist_resnet)

            _check_finite_actions(ref_actions, "Eager actor mean")
            _check_finite_actions(jit_actions, "Traced actor mean")
            max_abs_err = torch.max(torch.abs(ref_actions - jit_actions)).item()
            if max_abs_err > TRACE_MAX_ABS_TOLERANCE:
                raise RuntimeError(
                    "TorchScript trace differs from eager Actor: "
                    f"max_abs_err={max_abs_err:.8f}, tolerance={TRACE_MAX_ABS_TOLERANCE:.8f}"
                )
            if bounded_actor_mean:
                _check_action_bounds(ref_actions, "Eager actor mean")
                _check_action_bounds(jit_actions, "Traced actor mean")

        traced.save(output_path)
        reloaded = torch.jit.load(output_path, map_location="cpu")
        reloaded.eval()
        with torch.inference_mode():
            if args_cli.input_mode == "rgb":
                reloaded_actions = reloaded(probe_action_history, probe_proprio_obs, probe_wrist_rgb)
            else:
                reloaded_actions = reloaded(probe_action_history, probe_proprio_obs, probe_wrist_resnet)
            _check_finite_actions(reloaded_actions, "Reloaded Actor mean")
            reload_max_abs_err = torch.max(torch.abs(ref_actions - reloaded_actions)).item()
            if reload_max_abs_err > TRACE_MAX_ABS_TOLERANCE:
                raise RuntimeError(
                    "Reloaded TorchScript differs from eager Actor: "
                    f"max_abs_err={reload_max_abs_err:.8f}, tolerance={TRACE_MAX_ABS_TOLERANCE:.8f}"
                )
            if bounded_actor_mean:
                _check_action_bounds(reloaded_actions, "Reloaded Actor mean")
        if args_cli.input_mode == "rgb":
            embedded_encoder_hash = module_state_dict_sha256(reloaded.vision_encoder)
            expected_encoder_hash = encoder_manifest["state_dict_sha256"]
            if embedded_encoder_hash != expected_encoder_hash:
                raise RuntimeError(
                    "TorchScript embedded encoder hash mismatch: "
                    f"expected {expected_encoder_hash}, got {embedded_encoder_hash}"
                )
            encoder_manifest = dict(encoder_manifest)
            encoder_manifest["embedded_state_dict_sha256"] = embedded_encoder_hash
            encoder_manifest["embedded"] = True

        input_signature = {
            "action_history": [export_wrapper.action_history_dim],
            "proprio_obs": [export_wrapper.proprio_obs_dim],
        }
        notes = [
            (
                "Outputs tanh-bounded actor mean actions before environment-side action_scale, IK, and safety gating."
                if bounded_actor_mean
                else "Outputs actor mean actions before environment-side action_scale, IK, and safety gating."
            ),
            "critic_* privileged observations are internally zero-filled because the actor does not consume them.",
        ]
        if args_cli.input_mode == "rgb":
            input_signature["wrist_rgb"] = [rgb_height, rgb_width, 3]
            notes.append("wrist_rgb is expected as uint8 RGB frames in HWC layout.")
            notes.append("The export includes the same frozen ResNet18 ImageNet feature extractor used by the environment.")
        else:
            input_signature["wrist_resnet"] = [export_wrapper.wrist_resnet_dim]
            notes.append("This compact export expects precomputed wrist_resnet features instead of raw RGB frames.")

        metadata = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "checkpoint_sha256": sha256_file(resume_path),
            "output": output_path,
            "torchscript_sha256": sha256_file(output_path),
            "input_mode": args_cli.input_mode,
            "input_signature": input_signature,
            "output_signature": {
                "mean_actions": [policy.num_actions],
            },
            "action_semantics": ["dx", "dy", "dz", "gripper"],
            "actor_mean_transform": "tanh" if bounded_actor_mean else "identity",
            "actor_mean_bounds": [-1.0, 1.0] if bounded_actor_mean else None,
            "policy_contract": run_contract,
            "agent_config_source": str(checkpoint_cfg_path),
            "env_config_source": str(checkpoint_env_cfg_path),
            "action_history_semantics": "transition-associated environment-processed action after scale/clamp and before dz gate",
            "nominal_policy_frequency_hz": 1.0 / (float(env_cfg.sim.dt) * int(env_cfg.decimation)),
            "environment_action_scale": float(env_cfg.action_scale),
            "vision_encoder": encoder_manifest,
            "notes": notes,
            "trace_max_abs_err": max_abs_err,
            "reload_max_abs_err": reload_max_abs_err,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Exported TorchScript module to: {output_path}")
        print(f"[INFO] Export metadata written to: {metadata_path}")
        print(f"[INFO] Trace max abs error: {max_abs_err:.8f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
