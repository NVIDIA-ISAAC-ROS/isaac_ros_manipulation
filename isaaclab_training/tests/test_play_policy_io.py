# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for DisplayPort policy inference telemetry."""

import csv
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def play_policy_io(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["play_policy_io"])
    return importlib.import_module("isaaclab_training.cli.play_policy_io")


def _write_csv(path, rows: list[dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_task_space_logger_clips_raw_action_before_anisotropic_scale(
    play_policy_io,
):
    """Deploy-facing CSV actions must match the RSL-RL and LEAPP transform."""
    position_scale = (0.025, 0.020, 0.010)
    orientation_scale = (0.100, 0.200, 0.300)
    action_cfg = SimpleNamespace(position_scale=position_scale,
                                 orientation_scale=orientation_scale)
    robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos_target=torch.zeros((1, 7)),
        joint_names=[f"joint{i}" for i in range(1, 8)],
    ))
    base_env = SimpleNamespace(
        cfg=SimpleNamespace(actions=SimpleNamespace(arm_action=action_cfg), num_arm_joints=7),
        scene={"robot": robot},
    )
    env = SimpleNamespace(unwrapped=base_env)
    logger = play_policy_io.InferenceLogger(
        env,
        log_dir=None,
        print_enabled=False,
        print_every=1,
        clip_actions=1.0,
    )
    raw_action = np.array([2.0, -2.0, 0.5, -0.5, 1.5, -1.5])
    row = {f"action_raw_{i}": float(value) for i, value in enumerate(raw_action)}

    logger.end_step(row, success_info=None)

    logged_action = np.array([row[f"action_{i}"] for i in range(6)])
    np.testing.assert_allclose(logged_action, [0.025, -0.020, 0.005, -0.050, 0.200, -0.300])
    np.testing.assert_allclose(
        [row[f"action_raw_{i}"] for i in range(6)],
        raw_action,
    )


def test_task_space_replay_prefers_raw_actions_and_does_not_guess_scene_seed(
    play_policy_io, tmp_path
):
    csv_path = tmp_path / "task_space_raw.csv"
    raw = np.array([0.25, -0.5, 0.75, -1.0, 0.1, -0.2])
    row = {f"obs_{i}": float(10 + i) for i in range(18)}
    row.update({f"action_{i}": float(100 + i) for i in range(6)})
    row.update({f"action_raw_{i}": float(value) for i, value in enumerate(raw)})
    _write_csv(csv_path, [row])

    replay = play_policy_io.load_replay_trajectory(
        csv_path,
        task_space_scale=np.array([0.01, 0.02, 0.03, 0.1, 0.2, 0.3]),
        clip_actions=1.0,
    )

    assert replay.action_space == "task_space"
    assert replay.source == "action_raw (task-space)"
    np.testing.assert_allclose(replay.targets, raw[None, :])
    assert replay.init_joint_pos is None
    assert replay.observed_socket_pos is None
    assert replay.observed_socket_rot is None


def test_task_space_replay_recovers_raw_actions_from_scaled_actions(play_policy_io, tmp_path):
    csv_path = tmp_path / "task_space_scaled.csv"
    scale = np.array([0.01, 0.02, 0.03, 0.1, 0.2, 0.3])
    unclipped_raw = np.array([0.5, -2.0, 0.25, 1.5, -0.75, 0.1])
    clipped_raw = np.clip(unclipped_raw, -1.0, 1.0)
    row = {f"action_{i}": float(value) for i, value in enumerate(clipped_raw * scale)}
    _write_csv(csv_path, [row])

    replay = play_policy_io.load_replay_trajectory(
        csv_path, task_space_scale=scale, clip_actions=1.0
    )

    assert replay.source == "action (task-space scaled)"
    np.testing.assert_allclose(replay.targets, clipped_raw[None, :])


def test_joint_replay_still_loads_absolute_targets(play_policy_io, tmp_path):
    csv_path = tmp_path / "joint.csv"
    target = np.linspace(0.1, 0.7, 7)
    joints = np.linspace(-0.7, -0.1, 7)
    row = {f"target_joint_pos_{i}": float(value) for i, value in enumerate(target)}
    row.update({f"joint_joint{i + 1}_pos": float(value) for i, value in enumerate(joints)})
    _write_csv(csv_path, [row])

    replay = play_policy_io.load_replay_trajectory(csv_path)

    assert replay.action_space == "joint"
    np.testing.assert_allclose(replay.targets, target[None, :])
    np.testing.assert_allclose(replay.init_joint_pos, joints)


def test_task_space_init_obs_is_opaque_without_explicit_scene_columns(
    play_policy_io, tmp_path
):
    csv_path = tmp_path / "task_space_init.csv"
    obs = np.arange(18, dtype=np.float64)
    row = {f"obs_{i}": float(value) for i, value in enumerate(obs)}
    row.update({f"rnn_in_{i}": float(value) for i, value in enumerate([1.0, 2.0, 3.0])})
    _write_csv(csv_path, [row])

    seed = play_policy_io.load_init_policy_obs(csv_path, task_space=True)

    np.testing.assert_allclose(seed.obs, obs)
    np.testing.assert_allclose(seed.rnn_in, [1.0, 2.0, 3.0])
    assert seed.joint_pos is None
    assert seed.observed_socket_pos is None
    assert seed.observed_socket_rot is None


def test_task_space_init_obs_uses_explicit_scene_columns(play_policy_io, tmp_path):
    csv_path = tmp_path / "task_space_init_explicit.csv"
    joints = np.linspace(-0.3, 0.3, 7)
    row = {f"obs_{i}": float(i) for i in range(18)}
    row.update({f"joint_joint{i + 1}_pos": float(value) for i, value in enumerate(joints)})
    row.update(
        {
            "goal_px": 0.4,
            "goal_py": 0.5,
            "goal_pz": 0.6,
            "goal_qx": 0.0,
            "goal_qy": 0.0,
            "goal_qz": 0.0,
            "goal_qw": 1.0,
        }
    )
    _write_csv(csv_path, [row])

    seed = play_policy_io.load_init_policy_obs(csv_path, task_space=True)

    np.testing.assert_allclose(seed.joint_pos, joints)
    np.testing.assert_allclose(seed.observed_socket_pos, [0.4, 0.5, 0.6])
    np.testing.assert_allclose(seed.observed_socket_rot, [0.0, 0.0, 0.0, 1.0])


class _FakeRslPolicy:
    def __init__(self):
        self.rnn = SimpleNamespace(rnn=torch.nn.LSTM(4, 3, num_layers=2), hidden_state=None)

    def reset(self, dones=None, hidden_state=None):
        assert dones is None
        self.rnn.hidden_state = hidden_state

    def get_hidden_state(self):
        return self.rnn.hidden_state


def test_restore_rsl_rnn_state_validates_and_broadcasts_logged_state(play_policy_io):
    policy = _FakeRslPolicy()
    packed = np.arange(12, dtype=np.float64) + 0.25  # 2 layers x (h + c) x hidden 3

    play_policy_io._restore_rsl_rnn_state(policy, packed, num_envs=2)

    h, c = policy.get_hidden_state()
    assert h.shape == c.shape == (2, 2, 3)
    np.testing.assert_allclose(h[:, 0].detach().numpy(), [[0.25, 1.25, 2.25], [6.25, 7.25, 8.25]])
    np.testing.assert_allclose(
        c[:, 0].detach().numpy(), [[3.25, 4.25, 5.25], [9.25, 10.25, 11.25]]
    )
    np.testing.assert_allclose(h[:, 0], h[:, 1])
    np.testing.assert_allclose(c[:, 0], c[:, 1])

    with pytest.raises(ValueError, match="expected 12"):
        play_policy_io._restore_rsl_rnn_state(policy, packed[:-1], num_envs=2)


def test_restore_leapp_rnn_state_validates_and_broadcasts_feedback(play_policy_io):
    policy = object.__new__(play_policy_io.LeappDisplayportPolicy)
    policy.node_name = "actor"
    policy.inference = SimpleNamespace(
        value_dict={
            "actor": {
                "actor_state_0_in": torch.zeros((2, 2, 3)),
                "actor_state_1_in": torch.zeros((2, 2, 3)),
            }
        }
    )
    packed = np.arange(12, dtype=np.float64) + 0.5

    policy.restore_packed_rnn_state(packed)

    np.testing.assert_allclose(policy._read_feedback_packed(), packed)
    for state in policy.inference.value_dict["actor"].values():
        np.testing.assert_allclose(state[:, 0], state[:, 1])

    with pytest.raises(ValueError, match="expected 12"):
        policy.restore_packed_rnn_state(packed[:-1])
