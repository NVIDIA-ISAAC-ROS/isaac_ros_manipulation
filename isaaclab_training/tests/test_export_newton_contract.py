# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for DisplayPort task-space LEAPP observation contracts."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_export_module():
    """Load the standalone export contract without starting Isaac Sim."""
    return importlib.import_module("isaaclab_training.export.contracts")


def test_task_space_input_spec_defaults_to_physx_contract():
    """Test configs without order metadata retain the EEF-first contract."""
    export_module = _load_export_module()

    resolved = export_module.resolve_task_space_input_spec(SimpleNamespace())

    assert resolved is export_module._TASK_SPACE_INPUT_SPEC
    assert [entry[0] for entry in resolved] == [
        "eef_pos",
        "eef_rot_6d",
        "socket_kp_pos",
        "socket_kp_rot_6d",
    ]


def test_task_space_input_spec_uses_newton_observation_order():
    """Test Newton metadata exposes EEF-first ports mapped from the trained actor order."""
    export_module = _load_export_module()
    obs_order = ["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"]

    resolved = export_module.resolve_task_space_input_spec(
        SimpleNamespace(task_space_obs_order=obs_order))

    assert [entry[0] for entry in resolved] == [
        "eef_pos",
        "eef_rot_6d",
        "socket_kp_pos",
        "socket_kp_rot_6d",
    ]
    assert [(entry[1].start, entry[1].stop) for entry in resolved] == [(3, 6), (6, 12), (0, 3),
                                                                       (12, 18)]
    assert [entry[3] for entry in resolved] == [
        "eef_pose_pos",
        "eef_pose_rot6d",
        "socket_kp_pose_pos",
        "socket_kp_pose_rot6d",
    ]
    assert [entry[5] for entry in resolved] == [3, 6, 0, 12]


def test_task_space_input_terms_map_newton_aliases_to_canonical_ports():
    """Test runtime playback reads Newton aliases for canonical LEAPP inputs."""
    export_module = _load_export_module()
    env_cfg = SimpleNamespace(
        task_space_obs_order=["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"])

    resolved = export_module.resolve_task_space_input_terms(env_cfg)

    assert resolved == {
        "eef_pos": "tool_pos",
        "eef_rot_6d": "tool_rot_6d",
        "socket_kp_pos": "socket_pos",
        "socket_kp_rot_6d": "socket_rot_6d",
    }


def test_task_space_inputs_are_annotated_eef_first_and_rebuilt_in_newton_actor_order(monkeypatch):
    """Test public input order does not alter the vector consumed by a Newton checkpoint."""
    export_module = _load_export_module()
    env_cfg = SimpleNamespace(
        task_space_obs_order=["socket_pos", "tool_pos", "tool_rot_6d", "socket_rot_6d"],
        **_newton_deploy_metadata(),
    )
    resolved = export_module.resolve_task_space_input_spec(env_cfg)
    input_metadata, _ = export_module.resolve_task_space_deploy_metadata(env_cfg)
    annotation_order = []
    annotated_values = {}
    annotated_extras = {}

    def _annotate_input(_graph_name, semantics):
        annotation_order.append(semantics.name)
        annotated_values[semantics.name] = semantics.ref.clone()
        annotated_extras[semantics.name] = semantics.extra
        return semantics.ref

    leapp_module = ModuleType("leapp")
    leapp_module.__path__ = []
    leapp_utils_module = ModuleType("leapp.utils")
    leapp_utils_module.__path__ = []
    tensor_description_module = ModuleType("leapp.utils.tensor_description")
    tensor_description_module.TensorSemantics = SimpleNamespace
    monkeypatch.setitem(sys.modules, "leapp", leapp_module)
    monkeypatch.setitem(sys.modules, "leapp.utils", leapp_utils_module)
    monkeypatch.setitem(sys.modules, "leapp.utils.tensor_description", tensor_description_module)
    leapp_module.annotate = SimpleNamespace(input_tensors=_annotate_input)
    trained_actor_obs = torch.arange(18, dtype=torch.float32).reshape(1, 18)

    rebuilt = export_module.split_and_annotate_task_space_obs("DisplayPortTaskSpace",
                                                              trained_actor_obs,
                                                              input_spec=resolved,
                                                              input_metadata=input_metadata)

    assert annotation_order == ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]
    torch.testing.assert_close(annotated_values["eef_pos"], trained_actor_obs[:, 3:6])
    torch.testing.assert_close(annotated_values["eef_rot_6d"], trained_actor_obs[:, 6:12])
    torch.testing.assert_close(annotated_values["socket_kp_pos"], trained_actor_obs[:, 0:3])
    torch.testing.assert_close(annotated_values["socket_kp_rot_6d"], trained_actor_obs[:, 12:18])
    assert annotated_extras["eef_pos"]["policy_abi"] == "displayport_newton_flange_v1"
    assert annotated_extras["eef_pos"]["pose_reference"] == "flange"
    assert annotated_extras["eef_pos"]["body_offset"] == pytest.approx([0.0, 0.0, 0.0])
    assert "pose_reference" not in annotated_extras["socket_kp_pos"]
    torch.testing.assert_close(rebuilt, trained_actor_obs)


def _newton_deploy_metadata():
    return {
        "task_space_policy_abi": "displayport_newton_flange_v1",
        "task_space_backend": "newton_mjwarp",
        "task_space_observation_reference": "flange",
        "task_space_observation_body_name": "flange",
        "task_space_observation_body_offset": [0.0, 0.0, 0.0],
        "task_space_action_reference": "robot_root",
        "task_space_action_body_name": "flange",
        "task_space_action_body_offset": [0.0, 0.0, 0.0],
    }


def test_task_space_deploy_metadata_distinguishes_newton_frame():
    """Export metadata must make the Newton raw-flange ABI machine-readable."""
    export_module = _load_export_module()

    inputs, action = export_module.resolve_task_space_deploy_metadata(
        SimpleNamespace(**_newton_deploy_metadata()))

    assert inputs["eef_pos"] == {
        "policy_abi": "displayport_newton_flange_v1",
        "physics_backend": "newton_mjwarp",
        "coordinate_frame": "world",
        "pose_reference": "flange",
        "body_name": "flange",
        "body_offset": [0.0, 0.0, 0.0],
    }
    assert inputs["eef_rot_6d"] == inputs["eef_pos"]
    assert action == {
        "policy_abi": "displayport_newton_flange_v1",
        "physics_backend": "newton_mjwarp",
        "pose_reference": "robot_root",
        "body_name": "flange",
        "body_offset": [0.0, 0.0, 0.0],
    }


def test_task_space_deploy_metadata_rejects_partial_contract():
    """A partially declared frame contract must fail instead of exporting ambiguous inputs."""
    export_module = _load_export_module()

    with pytest.raises(ValueError, match="Incomplete task-space deployment metadata"):
        export_module.resolve_task_space_deploy_metadata(
            SimpleNamespace(task_space_policy_abi="displayport_newton_flange_v1"))


def test_export_task_space_action_preserves_legacy_metadata(monkeypatch):
    """Direct helper callers must retain the original Isaac Lab action metadata."""
    export_module = _load_export_module()
    captured = {}

    def annotate_output(_graph_name, semantics, export_with):
        captured["semantics"] = semantics
        captured["export_with"] = export_with

    leapp_module = ModuleType("leapp")
    leapp_module.__path__ = []
    leapp_utils_module = ModuleType("leapp.utils")
    leapp_utils_module.__path__ = []
    tensor_description_module = ModuleType("leapp.utils.tensor_description")
    tensor_description_module.TensorSemantics = SimpleNamespace
    monkeypatch.setitem(sys.modules, "leapp", leapp_module)
    monkeypatch.setitem(sys.modules, "leapp.utils", leapp_utils_module)
    monkeypatch.setitem(sys.modules, "leapp.utils.tensor_description", tensor_description_module)
    leapp_module.annotate = SimpleNamespace(output_tensors=annotate_output)

    export_module.export_task_space_action(
        "DisplayPortTaskSpace",
        torch.zeros(1, 6),
        "onnx",
    )

    assert captured["semantics"].extra == {
        "isaaclab_connection": "action:arm_action:pose_rel",
        "target_types": ["pose_rel"],
    }
    assert captured["export_with"] == "onnx"


def test_task_space_action_export_includes_scale_clip_and_frame(monkeypatch):
    """The custom LEAPP contract must preserve OSC transform and frame semantics."""
    export_module = _load_export_module()
    captured = {}

    def _annotate_output(_graph_name, semantics, export_with):
        captured["semantics"] = semantics
        captured["export_with"] = export_with

    leapp_module = ModuleType("leapp")
    leapp_module.__path__ = []
    leapp_utils_module = ModuleType("leapp.utils")
    leapp_utils_module.__path__ = []
    tensor_description_module = ModuleType("leapp.utils.tensor_description")
    tensor_description_module.TensorSemantics = SimpleNamespace
    monkeypatch.setitem(sys.modules, "leapp", leapp_module)
    monkeypatch.setitem(sys.modules, "leapp.utils", leapp_utils_module)
    monkeypatch.setitem(sys.modules, "leapp.utils.tensor_description", tensor_description_module)
    leapp_module.annotate = SimpleNamespace(output_tensors=_annotate_output)

    action_cfg = SimpleNamespace(
        position_scale=[0.025, 0.025, 0.010],
        orientation_scale=0.025,
        controller_cfg=SimpleNamespace(target_types=["pose_rel"]),
    )
    env_cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=action_cfg),
                              **_newton_deploy_metadata())
    contract = export_module.DisplayPortTaskSpaceContract()

    contract.export_action(
        contract.graph_name,
        torch.tensor([[2.0, 0.0, -0.5, 0.0, 0.0, 0.0]]),
        env_cfg,
        "cpu",
        torch.float32,
        "onnx",
        1.0,
    )

    semantics = captured["semantics"]
    torch.testing.assert_close(semantics.ref, torch.tensor([[0.025, 0.0, -0.005, 0.0, 0.0, 0.0]]))
    assert semantics.extra["scale"] == pytest.approx([0.025, 0.025, 0.010, 0.025, 0.025, 0.025])
    assert semantics.extra["position_scale"] == pytest.approx([0.025, 0.025, 0.010])
    assert semantics.extra["orientation_scale"] == pytest.approx(0.025)
    assert semantics.extra["clip_actions"] == pytest.approx(1.0)
    assert semantics.extra["policy_abi"] == "displayport_newton_flange_v1"
    assert semantics.extra["physics_backend"] == "newton_mjwarp"
    assert semantics.extra["pose_reference"] == "robot_root"
    assert semantics.extra["body_name"] == "flange"
    assert semantics.extra["body_offset"] == pytest.approx([0.0, 0.0, 0.0])
    assert captured["export_with"] == "onnx"


@pytest.mark.parametrize(
    ("obs_order", "message"),
    [
        pytest.param(
            ["socket_pos", "unknown", "tool_rot_6d", "socket_rot_6d"], "Unknown", id="unknown"),
        pytest.param(
            ["socket_pos", "tool_pos", "tool_rot_6d", "tool_rot_6d"], "Duplicate", id="duplicate"),
        pytest.param(["socket_pos", "tool_pos"], "exactly 18", id="wrong-width"),
        pytest.param(
            ["socket_pos", "socket_kp_pos", "tool_rot_6d", "socket_rot_6d"],
            "Duplicate Deploy input",
            id="aliased-input",
        ),
    ],
)
def test_task_space_input_spec_rejects_invalid_metadata(obs_order, message):
    """Test invalid observation ABI metadata fails before policy export."""
    export_module = _load_export_module()

    with pytest.raises(ValueError, match=message):
        export_module.resolve_task_space_input_spec(
            SimpleNamespace(task_space_obs_order=obs_order))


def test_task_space_action_matches_runner_clip_then_osc_scale():
    """Deployment must reproduce the model-999 raw-action transform."""
    export_module = _load_export_module()
    actions = torch.tensor([[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]])
    scale = torch.tensor([[0.025, 0.025, 0.01, 0.025, 0.025, 0.025]])

    processed = export_module.process_task_space_action(actions, scale, clip_actions=1.0)

    torch.testing.assert_close(
        processed,
        torch.tensor([[-0.025, -0.025, -0.005, 0.0125, 0.025, 0.025]]),
    )


def test_task_space_playback_inverts_deploy_transform_without_double_scaling():
    """Test LEAPP simulation playback recovers the runner action from a pose delta."""
    export_module = _load_export_module()
    raw_action = torch.tensor([[-2.0, -0.5, 0.25, 0.5, 1.0, 2.0]])
    scale = torch.tensor([[0.025, 0.025, 0.01, 0.025, 0.025, 0.025]])
    deploy_action = export_module.process_task_space_action(raw_action, scale, clip_actions=1.0)

    recovered = export_module.recover_task_space_policy_action(deploy_action,
                                                               scale,
                                                               clip_actions=1.0)

    torch.testing.assert_close(recovered, raw_action.clamp(-1.0, 1.0))


def test_task_space_playback_rejects_zero_action_scale():
    """Test invalid task-space scales fail instead of dividing by zero."""
    export_module = _load_export_module()

    with pytest.raises(ValueError, match="must be non-zero"):
        export_module.recover_task_space_policy_action(
            torch.ones(1, 6),
            torch.tensor([[0.025, 0.025, 0.0, 0.025, 0.025, 0.025]]),
            clip_actions=1.0,
        )


def test_task_space_action_scale_expands_scalar_and_per_axis_values():
    """Test OSC scales accept scalar and three-axis configuration values."""
    export_module = _load_export_module()
    env_cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(
        position_scale=[0.025, 0.025, 0.01],
        orientation_scale=0.025,
    )))

    scale = export_module.task_space_action_scale(env_cfg, "cpu", torch.float32)

    torch.testing.assert_close(scale, torch.tensor([[0.025, 0.025, 0.01, 0.025, 0.025, 0.025]]))


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        pytest.param("position_scale", [0.025, 0.025], id="position-two-axis"),
        pytest.param("orientation_scale", [0.025, 0.025, 0.025, 0.025],
                     id="orientation-four-axis"),
    ],
)
def test_task_space_action_scale_rejects_invalid_axis_count(attribute, value):
    """Test OSC scales reject sequences that are neither scalar nor three-axis."""
    export_module = _load_export_module()
    action_cfg = SimpleNamespace(position_scale=0.025, orientation_scale=0.025)
    setattr(action_cfg, attribute, value)
    env_cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=action_cfg))

    with pytest.raises(ValueError, match=f"{attribute} must contain either one or three values"):
        export_module.task_space_action_scale(env_cfg, "cpu", torch.float32)


def test_task_space_action_scale_rejects_non_finite_values():
    """Test OSC scales reject non-finite deployment transforms."""
    export_module = _load_export_module()
    env_cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(
        position_scale=[0.025, float("nan"), 0.01], orientation_scale=0.025)))

    with pytest.raises(ValueError, match="position_scale must contain only finite values"):
        export_module.task_space_action_scale(env_cfg, "cpu", torch.float32)


def test_task_space_action_can_leave_runner_action_unclipped():
    """A disabled runner clip must not add an exporter-only clamp."""
    export_module = _load_export_module()
    actions = torch.tensor([[-2.0, 2.0]])

    processed = export_module.process_task_space_action(actions,
                                                        torch.full((1, 2), 0.025),
                                                        clip_actions=None)

    torch.testing.assert_close(processed, torch.tensor([[-0.05, 0.05]]))
