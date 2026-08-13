"""Contract tests for the three-input direct-action RMA Student."""

from __future__ import annotations

import sys

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import pytest
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_direct_action_student.artifacts import (
    RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_KIND,
    RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_VERSION,
    RMA_DIRECT_ACTION_STUDENT_DR_TASK,
    direct_action_input_contract,
    load_student_checkpoint,
    state_dict_sha256,
)
from tacex_tasks.sim2real_grasp.rma_direct_action_student.models import (RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION, RMADirectActionVisualStudent)
from tacex_tasks.sim2real_grasp.rma_xy_artifacts import RMA_XY_STUDENT_HEATMAP_DR_TASK


@pytest.fixture(scope="module", autouse=True)
def close_simulation_app():
    yield
    simulation_app.close()


def test_direct_action_model_is_three_input_and_finetunes_late_backbone():
    student = RMADirectActionVisualStudent().train()
    inputs = (
        torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((2, 15), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
    )
    inputs[1][:, -1] = 0.04
    actions = student(*inputs)
    assert actions.shape == (2, 4)
    assert torch.all(actions.abs() <= 1.0)
    assert student.contract()["runtime_privileged_inputs"] == []
    assert student.contract()["input_order"] == ["wrist_rgb", "proprio_obs", "action_history"]
    actions.square().mean().backward()
    assert all(parameter.grad is None for index, module in enumerate(student.vision_encoder) if index < 6 for parameter in module.parameters())
    assert any(parameter.grad is not None for index, module in enumerate(student.vision_encoder) if index >= 6 for parameter in module.parameters())


def test_direct_action_torchscript_has_exactly_three_inputs(tmp_path):
    student = RMADirectActionVisualStudent().eval()
    inputs = (
        torch.zeros((2, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((2, 15), dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
    )
    traced = torch.jit.script(student)
    with torch.inference_mode():
        torch.testing.assert_close(student(*inputs), traced(*inputs))
        assert traced(*(value[:1] for value in inputs)).shape == (1, 4)
    export = tmp_path / "student.pt"
    traced.save(str(export))
    with torch.inference_mode():
        torch.testing.assert_close(
            traced(*inputs), torch.jit.load(str(export), map_location="cpu")(*inputs)
        )


def test_direct_action_artifact_rejects_old_xy_student(tmp_path):
    checkpoint = tmp_path / "old_xy_student.pt"
    torch.save({"kind": "tacex_rma_xy_student", "version": 2}, checkpoint)
    with pytest.raises(RuntimeError, match="not an RMA direct-action Student"):
        load_student_checkpoint(checkpoint)
    assert direct_action_input_contract() == {
        "wrist_rgb": [224, 224, 3], "proprio_obs": [15], "action_history": [4]
    }


def test_direct_action_artifact_requires_encoder_provenance_and_hash(tmp_path):
    student = RMADirectActionVisualStudent()
    payload = {
        "kind": RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_KIND,
        "version": RMA_DIRECT_ACTION_STUDENT_CHECKPOINT_VERSION,
        "model_version": RMA_DIRECT_ACTION_STUDENT_MODEL_VERSION,
        "task": RMA_DIRECT_ACTION_STUDENT_DR_TASK,
        "model": student.state_dict(),
        "student_input_contract": direct_action_input_contract(),
        "student_model_contract": student.contract(),
        "normalization": student.normalizer.contract(),
        "encoder_init_task": RMA_XY_STUDENT_HEATMAP_DR_TASK,
        "encoder_init_checkpoint_sha256": "a" * 64,
        "encoder_init_state_dict_sha256": "b" * 64,
        "vision_encoder_state_dict_sha256": state_dict_sha256(student.vision_encoder.state_dict()),
    }
    checkpoint = tmp_path / "direct.pt"
    torch.save(payload, checkpoint)
    assert load_student_checkpoint(checkpoint)["task"] == RMA_DIRECT_ACTION_STUDENT_DR_TASK
    payload.pop("encoder_init_checkpoint_sha256")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="encoder initialization provenance"):
        load_student_checkpoint(checkpoint)
    payload["encoder_init_checkpoint_sha256"] = "a" * 64
    payload["vision_encoder_state_dict_sha256"] = "wrong"
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="vision encoder state hash"):
        load_student_checkpoint(checkpoint)


def test_direct_action_dr_task_is_registered_and_keeps_training_labels_only():
    assert gym.spec(RMA_DIRECT_ACTION_STUDENT_DR_TASK) is not None
    cfg = parse_env_cfg(RMA_DIRECT_ACTION_STUDENT_DR_TASK, device="cuda:0", num_envs=1)
    assert cfg.wrist_visual_randomization_enabled is True
    assert cfg.dr_curriculum_enabled is False
    # Labels remain available to the asymmetric trainer, not in the exported
    # Student input contract asserted above.
    assert cfg.observation_space["rma_cube_xy"] == 2
    assert cfg.observation_space["rma_contact_force"] == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
