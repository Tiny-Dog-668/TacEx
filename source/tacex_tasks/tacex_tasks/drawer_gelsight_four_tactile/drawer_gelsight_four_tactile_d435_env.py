"""D435-calibrated third view for the existing four-GelSight drawer task."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from tacex_tasks.third_view_grasping.vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)


@configclass
class DrawerGelSightFourTactileD435Cfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Keep the drawer task contract while using the Real-Alignment D435 view."""

    alignment_reference_id = "20260711_214450_real_alignment_reference"
    camera_extrinsics_source = "user_provided_base_T_camera_color_optical_20260726"
    camera_intrinsics_source = "20260726_194811_d435_crop_samples"
    deployment_camera_serial = "215322076207"
    camera_raw_resolution = (640, 480)
    camera_raw_intrinsic_matrix = (
        604.897400,
        0.0,
        320.980103,
        0.0,
        605.085815,
        247.913223,
        0.0,
        0.0,
        1.0,
    )
    camera_crop_roi_xywh = (100, 34, 400, 398)
    camera_model_intrinsic_matrix = (
        338.742544,
        0.0,
        123.748857,
        0.0,
        340.550811,
        120.393372,
        0.0,
        0.0,
        1.0,
    )
    # Omniverse renders centered square pixels. This wider native K is warped
    # before ResNet so the encoded image follows ``camera_model_intrinsic_matrix``.
    camera_native_render_intrinsic_matrix = (
        300.0,
        0.0,
        112.0,
        0.0,
        300.0,
        112.0,
        0.0,
        0.0,
        1.0,
    )
    camera_nominal_intrinsic_compensation_enabled = True
    camera_nominal_intrinsic_compensation_mode = (
        "gpu_affine_grid_centered_square_pixel_render_to_calibrated_k"
    )
    camera_base_position_m = (1.166091088407, 0.035901608197, 0.514200335898)
    camera_world_position_m = (1.166091088407, 0.035901608197, 0.534200335898)

    third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=1.0 / 30.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=list(camera_native_render_intrinsic_matrix),
            width=224,
            height=224,
            focal_length=1.0,
            focus_distance=0.8,
            clipping_range=(0.05, 30.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # Calibrated base_T_camera_color_optical, stored as Isaac wxyz
            # after converting the supplied ROS xyzw quaternion.
            pos=camera_world_position_m,
            rot=(0.378248136306, -0.604227000834, -0.586824979121, 0.384024117374),
            convention="ros",
        ),
    )


class DrawerGelSightFourTactileD435Env(OccludedGraspingVisionFourTactileBoxEnv):
    """Four-GelSight drawer environment with calibrated D435 visual input."""

    cfg: DrawerGelSightFourTactileD435Cfg

    def __init__(self, cfg: DrawerGelSightFourTactileD435Cfg, *args, **kwargs):
        self._d435_intrinsic_grid: torch.Tensor | None = None
        self._d435_intrinsic_grid_key: tuple[object, ...] | None = None
        super().__init__(cfg, *args, **kwargs)

    def _preprocess_third_person_rgb(self, rgb_nchw: torch.Tensor) -> torch.Tensor:
        """Map native 224-square RGB ``[N, 3, 224, 224]`` to calibrated D435 K."""
        if not bool(self.cfg.camera_nominal_intrinsic_compensation_enabled):
            return rgb_nchw

        batch_size, _, height, width = rgb_nchw.shape
        expected_size = (
            int(self.cfg.third_person_camera.width),
            int(self.cfg.third_person_camera.height),
        )
        if (width, height) != expected_size:
            raise ValueError(
                "D435 intrinsic compensation expects third-person camera resolution "
                f"{expected_size}, got {(width, height)}."
            )

        target = self.cfg.camera_model_intrinsic_matrix
        native = self.cfg.camera_native_render_intrinsic_matrix
        scale_x = float(native[0]) / float(target[0])
        scale_y = float(native[4]) / float(target[4])
        # ``affine_grid(..., align_corners=False)`` maps output pixel centers
        # to input coordinates. The half-pixel terms preserve the calibrated
        # principal point when native and target focal lengths differ.
        translate_x = (
            2.0
            * (float(native[2]) + 0.5 - scale_x * (float(target[2]) + 0.5))
            / float(width)
            - 1.0
            + scale_x
        )
        translate_y = (
            2.0
            * (float(native[5]) + 0.5 - scale_y * (float(target[5]) + 0.5))
            / float(height)
            - 1.0
            + scale_y
        )

        cache_key = (height, width, rgb_nchw.device, rgb_nchw.dtype)
        if self._d435_intrinsic_grid_key != cache_key:
            theta = torch.tensor(
                [[scale_x, 0.0, translate_x], [0.0, scale_y, translate_y]],
                device=rgb_nchw.device,
                dtype=rgb_nchw.dtype,
            ).unsqueeze(0)
            self._d435_intrinsic_grid = F.affine_grid(
                theta,
                size=(1, rgb_nchw.shape[1], height, width),
                align_corners=False,
            )
            self._d435_intrinsic_grid_key = cache_key

        assert self._d435_intrinsic_grid is not None
        return F.grid_sample(
            rgb_nchw,
            self._d435_intrinsic_grid.expand(batch_size, -1, -1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
