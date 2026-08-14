"""X040-Wide RMA tasks with eight reset-selectable physical cube sizes."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from isaaclab.assets import RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_rma_x040_wide_env import (
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnv,
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMAX040WideTeacherEnv,
    Sim2RealCubeRealAlignmentRMAX040WideTeacherEnvCfg,
)


_CUBE_SIZE_BUCKETS_M = tuple(0.04 + index * (0.02 / 7.0) for index in range(8))


class _SelectedBucketContactData:
    """Gather the active bucket from eight one-cube/two-finger sensors."""

    def __init__(self, owner: "_SelectedBucketContactSensorBank") -> None:
        self._owner = owner

    @property
    def force_matrix_w_history(self) -> torch.Tensor:
        selected = None
        for bucket_id, sensor in enumerate(self._owner.sensors):
            history = sensor.data.force_matrix_w_history
            if history is None:
                raise RuntimeError("Size-bucket cube contact sensor has no force history")
            if selected is None:
                selected = torch.empty_like(history)
            selected[self._owner.active_bucket_ids == bucket_id] = history[
                self._owner.active_bucket_ids == bucket_id
            ]
        if selected is None:
            raise RuntimeError("Size-bucket contact sensor bank is empty")
        # Preserve the inherited contract [N,H,cube_body=1,fingers=2,xyz=3].
        return selected


class _SelectedBucketContactSensorBank:
    """Compatibility facade selecting one of eight cube contact sensors."""

    def __init__(
        self,
        sensors: list[ContactSensor],
        active_bucket_ids: torch.Tensor,
    ) -> None:
        self.sensors = sensors
        self.active_bucket_ids = active_bucket_ids
        self.data = _SelectedBucketContactData(self)
        self.num_bodies = 1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for sensor in self.sensors:
            sensor.reset(env_ids)


class _SelectedCubeData:
    """Expose the selected object in a collection as one rigid-object data view.

    CPU PhysX can return collection state on CPU while the DirectRLEnv policy
    tensors remain on CUDA.  This facade is the collection-to-environment
    boundary, so selected state is always returned on the environment device.
    """

    def __init__(self, owner: "_SelectedCube") -> None:
        self._owner = owner

    def _selected(self, values: torch.Tensor) -> torch.Tensor:
        rows = torch.arange(self._owner.num_envs, device=values.device)
        bucket_ids = self._owner.active_bucket_ids.to(device=values.device)
        selected = values[rows, bucket_ids]
        return selected.to(device=self._owner.environment_device, non_blocking=True)

    @property
    def default_root_state(self) -> torch.Tensor:
        state = self._selected(self._owner.collection.data.default_object_state).clone()
        state[:, 0] = float(self._owner.nominal_xy[0])
        state[:, 1] = float(self._owner.nominal_xy[1])
        return state

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._selected(self._owner.collection.data.object_pos_w)

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._selected(self._owner.collection.data.object_quat_w)

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._selected(self._owner.collection.data.object_lin_vel_w)

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._selected(self._owner.collection.data.object_ang_vel_w)


class _SelectedCube:
    """Route legacy single-cube reads/writes to one selected size bucket per env."""

    def __init__(
        self,
        collection: RigidObjectCollection,
        active_bucket_ids: torch.Tensor,
        env_origins: torch.Tensor,
        nominal_xy: tuple[float, float],
        environment_device: torch.device,
    ) -> None:
        self.collection = collection
        self.active_bucket_ids = active_bucket_ids
        self.env_origins = env_origins
        self.nominal_xy = nominal_xy
        self.environment_device = environment_device
        self.num_envs = int(active_bucket_ids.shape[0])
        self.data = _SelectedCubeData(self)

    def write_root_state_to_sim(
        self,
        root_state: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.active_bucket_ids.device)
        else:
            env_ids = env_ids.to(device=self.active_bucket_ids.device, dtype=torch.long)

        # Input root_state is [K,13] in world coordinates. Reset every bucket
        # to its separate behind-camera parking pose, then overwrite [K,1,13]
        # at the currently selected bucket with the active task pose.
        parked = self.collection.data.default_object_state[env_ids].clone()
        # Input parked/origins: [K,B,13] / [K,3]. CPU PhysX can leave scene
        # origins on CPU even though the collection state is on CUDA.
        env_origins = self.env_origins.to(
            device=parked.device,
            dtype=parked.dtype,
        )
        parked[..., :3] += env_origins[env_ids].unsqueeze(1)
        rows = torch.arange(env_ids.numel(), device=env_ids.device)
        parked[rows, self.active_bucket_ids[env_ids]] = root_state

        # This task intentionally runs its PhysX rigid-body solver with GPU
        # dynamics disabled (see ``_configure_size_bucket_physics``).  Isaac
        # Sim 4.5's Python tensor view otherwise still reaches
        # PxRigidDynamic::set*Velocity, which is forbidden once the scene has
        # eENABLE_DIRECT_GPU_API enabled.  The collection writer keeps Isaac
        # Lab's state caches and the complete [env, bucket, 13] reset atomic.
        self.collection.write_object_state_to_sim(parked, env_ids=env_ids)


class _CubeSizeBucketMixin:
    """Select a cube size and perturb the seven arm joints at every reset."""

    cfg: SimpleNamespace

    def _setup_scene(self) -> None:
        self._configure_size_bucket_physics()
        super()._setup_scene()
        # The inherited sim2real scene uses heterogeneous cloning
        # (replicate_physics=False), so Isaac cannot auto-filter environments.
        # Explicit isolation is required now that every env owns eight parked
        # dynamic cubes; the global ground remains collidable by every env.
        self.scene.filter_collisions(global_prim_paths=[self.cfg.ground.prim_path])

    def _configure_size_bucket_physics(self) -> None:
        """Select the solver mode before the eight dynamic Cube actors exist."""
        if "cuda" not in str(self.device):
            return

        physics_context = self.sim.get_physics_context()
        if not hasattr(physics_context, "enable_gpu_dynamics"):
            raise RuntimeError(
                "Size-Buckets requires PhysicsContext.enable_gpu_dynamics() to "
                "avoid incompatible Direct GPU rigid-body reset calls."
            )

        enable_gpu_dynamics = bool(self.cfg.enable_gpu_dynamics)
        physics_context.enable_gpu_dynamics(enable_gpu_dynamics)
        mode = "enabled" if enable_gpu_dynamics else "disabled"
        print(
            f"[INFO] Size-Buckets PhysX GPU dynamics {mode}; CUDA policy and rendering remain enabled.",
            flush=True,
        )

    def _setup_cube_target(self) -> None:
        bucket_cfgs = {}
        parking_xy = tuple(self.cfg.cube_bucket_parking_xy_m)
        if len(parking_xy) != len(self.cfg.cube_size_buckets_m):
            raise ValueError("cube_bucket_parking_xy_m must contain one pose per size bucket")

        # All cameras share the same local pose and look predominantly toward
        # negative X. A fixed local parking X can land beside a camera in the
        # next env row after adding env origins. Shift the complete parking bank
        # beyond the positive-X extent of the grid, then add an explicit margin
        # behind the rearmost camera. This remains safe as num_envs changes.
        origin_x_span = float(
            (self.scene.env_origins[:, 0].max() - self.scene.env_origins[:, 0].min()).item()
        )
        camera_local_x = float(self.cfg.wrist_camera.offset.pos[0])
        camera_position_delta = getattr(
            self.cfg, "camera_position_delta_max_m", (0.0, 0.0, 0.0)
        )
        camera_delta_x = abs(float(camera_position_delta[0]))
        configured_x_min = min(float(position[0]) for position in parking_xy)
        parking_x_base = (
            origin_x_span
            + camera_local_x
            + camera_delta_x
            + float(self.cfg.cube_bucket_parking_camera_margin_m)
        )
        self._cube_bucket_parking_local_xy_m = tuple(
            (parking_x_base + float(x) - configured_x_min, float(y)) for x, y in parking_xy
        )

        for bucket_id, (size_m, (parking_x, parking_y)) in enumerate(
            zip(self.cfg.cube_size_buckets_m, self._cube_bucket_parking_local_xy_m)
        ):
            cube_cfg = self.cfg.cube.copy()
            cube_cfg.prim_path = f"/World/envs/env_.*/cube_size_bucket_{bucket_id}"
            cube_cfg.spawn = cube_cfg.spawn.copy()
            cube_cfg.spawn.size = (float(size_m),) * 3
            cube_cfg.init_state = cube_cfg.init_state.copy()
            cube_cfg.init_state.pos = (
                float(parking_x),
                float(parking_y),
                float(self.cfg.plate_top_height_m) + 0.5 * float(size_m),
            )
            bucket_cfgs[f"cube_size_bucket_{bucket_id}"] = cube_cfg

        collection_cfg = RigidObjectCollectionCfg(rigid_objects=bucket_cfgs)
        self._cube_size_bucket_collection = RigidObjectCollection(collection_cfg)
        self.scene.rigid_object_collections["cube_size_buckets"] = self._cube_size_bucket_collection
        self._active_cube_bucket_ids = torch.randint(
            low=0,
            high=len(self.cfg.cube_size_buckets_m),
            size=(self.num_envs,),
        )
        self._cube_size_bucket_values = torch.tensor(
            self.cfg.cube_size_buckets_m,
            device=self.device,
            dtype=torch.float32,
        )
        selected_cube = _SelectedCube(
            collection=self._cube_size_bucket_collection,
            active_bucket_ids=self._active_cube_bucket_ids,
            env_origins=self.scene.env_origins,
            nominal_xy=(float(self.cfg.cube.init_state.pos[0]), float(self.cfg.cube.init_state.pos[1])),
            environment_device=self.device,
        )
        self._cube = selected_cube
        self._cylinder = selected_cube

    @property
    def active_cube_size_m(self) -> torch.Tensor:
        """Per-environment selected physical edge length, shape [N]."""
        return self._cube_size_bucket_values[self._active_cube_bucket_ids]

    def _setup_rma_cube_contact_sensor(self) -> None:
        sensors = []
        for bucket_id in range(len(self.cfg.cube_size_buckets_m)):
            sensor_cfg = self.cfg.rma_cube_contact_sensor.copy()
            sensor_cfg.prim_path = f"/World/envs/env_.*/cube_size_bucket_{bucket_id}"
            sensor = ContactSensor(sensor_cfg)
            sensors.append(sensor)
            self.scene.sensors[f"rma_cube_contact_sensor_bucket_{bucket_id}"] = sensor
        self.rma_cube_contact_sensor = _SelectedBucketContactSensorBank(
            sensors,
            self._active_cube_bucket_ids,
        )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._active_cube_bucket_ids[env_ids] = torch.randint(
            low=0,
            high=len(self.cfg.cube_size_buckets_m),
            size=(env_ids.numel(),),
            device=self.device,
        )
        super()._reset_idx(env_ids)

        # Sample [K,7] arm-joint offsets in radians. The fingers intentionally
        # retain the measured real opening, and joint velocities remain zero.
        std_rad = float(self.cfg.arm_joint_reset_noise_std_rad)
        clip_rad = float(self.cfg.arm_joint_reset_noise_clip_rad)
        if std_rad < 0.0 or clip_rad < 0.0:
            raise ValueError("Arm-joint reset noise std/clip must be non-negative")
        noise = torch.randn(
            (env_ids.numel(), 7),
            device=self.device,
            dtype=self._robot.data.default_joint_pos.dtype,
        ) * std_rad
        noise.clamp_(min=-clip_rad, max=clip_rad)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        arm_lower = self._robot.data.soft_joint_pos_limits[env_ids, :7, 0]
        arm_upper = self._robot.data.soft_joint_pos_limits[env_ids, :7, 1]
        noisy_arm_pos = torch.clamp(joint_pos[:, :7] + noise, min=arm_lower, max=arm_upper)
        joint_pos[:, :7] = noisy_arm_pos
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if not hasattr(self, "_last_arm_joint_reset_noise_rad"):
            self._last_arm_joint_reset_noise_rad = torch.zeros(
                (self.num_envs, 7),
                device=self.device,
                dtype=joint_pos.dtype,
            )
        # Record the applied rather than requested offset in case a limit clips it.
        self._last_arm_joint_reset_noise_rad[env_ids] = (
            noisy_arm_pos - self._robot.data.default_joint_pos[env_ids, :7]
        )

    def _compute_cube_lowest_height(
        self,
        cube_pos: torch.Tensor,
        cube_quat: torch.Tensor,
    ) -> torch.Tensor:
        """Return the selected cube's lowest world corner for [N,3]/[N,4]."""
        if cube_pos.shape[0] != self.num_envs:
            raise ValueError("Size-bucket cube geometry expects one pose per environment")
        unit_corners = torch.tensor(
            [
                [-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5],
                [0.5, -0.5, -0.5], [0.5, -0.5, 0.5],
                [0.5, 0.5, -0.5], [0.5, 0.5, 0.5],
            ],
            device=cube_pos.device,
            dtype=cube_pos.dtype,
        )
        # [N,8,3] local cube corners -> [N*8,3] quaternion rotation -> [N,8,3].
        local_corners = unit_corners.unsqueeze(0) * self.active_cube_size_m[:, None, None]
        env_corners = local_corners.reshape(-1, 3)
        env_quats = cube_quat.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)
        from isaaclab.utils import math as math_utils

        rotated = math_utils.quat_apply(env_quats, env_corners).view(self.num_envs, 8, 3)
        return (rotated + cube_pos.unsqueeze(1))[:, :, 2].amin(dim=1)

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        log["info/cube_size_mean_m"] = self.active_cube_size_m.mean().detach()
        log["info/cube_size_min_m"] = self.active_cube_size_m.min().detach()
        log["info/cube_size_max_m"] = self.active_cube_size_m.max().detach()
        if hasattr(self, "_last_arm_joint_reset_noise_rad"):
            reset_noise_abs = self._last_arm_joint_reset_noise_rad.abs()
            log["info/arm_joint_reset_noise_abs_mean_rad"] = reset_noise_abs.mean().detach()
            log["info/arm_joint_reset_noise_abs_max_rad"] = reset_noise_abs.max().detach()
        return reward, log


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsTeacherEnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideTeacherEnvCfg
):
    """Teacher with eight physical cube edge lengths sampled per reset."""

    cube_size_buckets_m = _CUBE_SIZE_BUCKETS_M
    cube_size_sampling = "uniform_discrete_per_environment_per_reset"
    # Isaac Sim 4.5 routes Python rigid-body reset writes through CPU
    # PxRigidDynamic setters when this is True, conflicting with its Direct
    # GPU API scene mode. Keep inference/rendering on CUDA while solving this
    # task's rigid-body physics on CPU.
    enable_gpu_dynamics = False
    cube_bucket_parking_strategy = "beyond_positive_x_env_extent_behind_all_cameras"
    cube_bucket_parking_camera_margin_m = 0.5
    arm_joint_reset_noise_distribution = "normal_clipped_per_environment_per_reset"
    arm_joint_reset_noise_std_rad = 0.01
    arm_joint_reset_noise_clip_rad = 0.03
    # X values encode only spacing within the bank; _setup_cube_target shifts
    # the complete bank behind every camera after env origins are known.
    cube_bucket_parking_xy_m = tuple(
        (1.35 + 0.09 * (index % 4), -0.14 + 0.28 * (index // 4))
        for index in range(8)
    )


class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsTeacherEnv(
    _CubeSizeBucketMixin,
    Sim2RealCubeRealAlignmentRMAX040WideTeacherEnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsTeacherEnvCfg


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg
):
    """Visual Student DR task matching the Teacher's size-bucket physics."""

    cube_size_buckets_m = _CUBE_SIZE_BUCKETS_M
    cube_size_sampling = "uniform_discrete_per_environment_per_reset"
    enable_gpu_dynamics = False
    cube_bucket_parking_strategy = "beyond_positive_x_env_extent_behind_all_cameras"
    cube_bucket_parking_camera_margin_m = 0.5
    arm_joint_reset_noise_distribution = "normal_clipped_per_environment_per_reset"
    arm_joint_reset_noise_std_rad = 0.01
    arm_joint_reset_noise_clip_rad = 0.03
    # X values encode only spacing within the bank; _setup_cube_target shifts
    # the complete bank behind every camera after env origins are known.
    cube_bucket_parking_xy_m = tuple(
        (1.35 + 0.09 * (index % 4), -0.14 + 0.28 * (index // 4))
        for index in range(8)
    )


class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnv(
    _CubeSizeBucketMixin,
    Sim2RealCubeRealAlignmentRMAX040WideStudentDREnv,
):
    cfg: Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg


__all__ = (
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsTeacherEnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsTeacherEnv",
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsStudentDREnv",
)
