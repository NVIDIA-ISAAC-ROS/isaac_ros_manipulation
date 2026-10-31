# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the RSL-RL 5 recurrent runner configuration."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import MISSING
import io
from pathlib import Path
import re
import warnings

from isaaclab_rl.rsl_rl import (handle_deprecated_rsl_rl_cfg, RslRlMLPModelCfg,
                                RslRlPpoActorCriticRecurrentCfg, RslRlRNNModelCfg)
from isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s \
    .agents.rsl_rl_ppo_cfg import \
    Rizon4sGravDisplayportInsertionRNNPPORunnerCfg
from rsl_rl.models import RNNModel
from tensordict import TensorDict
import torch

_RSL_RL_VERSION = "5.4.1"
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# The canonical training guides live in the Isaac ROS Manipulation documentation. Keep command
# validation pointed at that single source instead of requiring a second package-level README.
_TRAINING_DOCS = _PACKAGE_ROOT.parents[3] / "docs/isaac_ros_docs/manipulator_docs/training"
_DOCUMENTATION_PATHS = tuple(
    _TRAINING_DOCS / relative_path for relative_path in (
        "index.rst",
        "setup_training.md",
        "tasks/displayport_insertion.rst",
    ) if _TRAINING_DOCS.is_dir())
_TRAIN_COMMAND_START = re.compile(
    r"^(?:uv\s+run(?:\s+\S+)*?\s+isaaclab|(?:\./)?isaaclab\.sh)\s+train\b")
_NUM_ENVS = re.compile(r"--num_envs(?:=|\s+)(\d+)\b")


def _legacy_runner_cfg() -> Rizon4sGravDisplayportInsertionRNNPPORunnerCfg:
    """Create the pre-migration runner configuration for an equivalence check."""
    cfg = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg()
    cfg.actor = MISSING
    cfg.critic = MISSING
    cfg.policy = RslRlPpoActorCriticRecurrentCfg(
        state_dependent_std=True,
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        noise_std_type="log",
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    return cfg


def _rnn_model_from_cfg(
    cfg: RslRlRNNModelCfg,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
) -> RNNModel:
    """Instantiate an RSL-RL recurrent model from its Isaac Lab configuration."""
    kwargs = deepcopy(cfg.to_dict())
    assert kwargs.pop("class_name") == "RNNModel"
    return RNNModel(obs=obs,
                    obs_groups=obs_groups,
                    obs_set=obs_set,
                    output_dim=output_dim,
                    **kwargs)


def _documented_train_commands(path: Path) -> list[tuple[int, str]]:
    """Extract continued shell commands that invoke the Isaac Lab trainer."""
    lines = path.read_text(encoding="utf-8").splitlines()
    commands: list[tuple[int, str]] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        if not _TRAIN_COMMAND_START.match(line):
            line_index += 1
            continue

        start_line = line_index + 1
        command_parts = [line.removesuffix("\\").strip()]
        while line.endswith("\\") and line_index + 1 < len(lines):
            line_index += 1
            line = lines[line_index].strip()
            command_parts.append(line.removesuffix("\\").strip())
        commands.append((start_line, " ".join(command_parts)))
        line_index += 1
    return commands


def test_runner_uses_warning_free_rsl_rl_5_recurrent_models(capsys) -> None:
    """The runner must use the explicit RSL-RL 5 actor, critic, and distribution API."""
    cfg = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg()

    assert isinstance(cfg.actor, RslRlRNNModelCfg)
    assert cfg.actor.hidden_dims == [256, 128, 64]
    assert cfg.actor.activation == "elu"
    assert cfg.actor.obs_normalization is True
    assert cfg.actor.rnn_type == "lstm"
    assert cfg.actor.rnn_hidden_dim == 256
    assert cfg.actor.rnn_num_layers == 2
    assert isinstance(cfg.actor.distribution_cfg,
                      RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)
    assert cfg.actor.distribution_cfg.init_std == 1.0
    assert cfg.actor.distribution_cfg.std_type == "log"

    assert isinstance(cfg.critic, RslRlRNNModelCfg)
    assert cfg.critic.hidden_dims == [256, 128, 64]
    assert cfg.critic.activation == "elu"
    assert cfg.critic.obs_normalization is True
    assert cfg.critic.rnn_type == "lstm"
    assert cfg.critic.rnn_hidden_dim == 256
    assert cfg.critic.rnn_num_layers == 2
    assert cfg.critic.distribution_cfg is None
    assert isinstance(cfg.policy, type(MISSING))

    capsys.readouterr()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        migrated = handle_deprecated_rsl_rl_cfg(cfg, _RSL_RL_VERSION)
    captured = capsys.readouterr()

    assert migrated is cfg
    assert captured.out == ""
    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


def test_runner_migration_preserves_legacy_model_and_checkpoint_shapes() -> None:
    """The explicit RSL-RL 5 models must match the former compatibility conversion exactly."""
    explicit_cfg = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg()
    handle_deprecated_rsl_rl_cfg(explicit_cfg, _RSL_RL_VERSION)

    legacy_cfg = _legacy_runner_cfg()
    with redirect_stdout(io.StringIO()):
        handle_deprecated_rsl_rl_cfg(legacy_cfg, _RSL_RL_VERSION)

    assert explicit_cfg.actor.to_dict() == legacy_cfg.actor.to_dict()
    assert explicit_cfg.critic.to_dict() == legacy_cfg.critic.to_dict()

    obs = TensorDict(
        {
            "policy": torch.zeros(2, 18),
            "critic": torch.zeros(2, 40),
        },
        batch_size=[2],
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    for model_name, obs_set, output_dim in (("actor", "actor", 6), ("critic", "critic", 1)):
        torch.manual_seed(123)
        explicit_model = _rnn_model_from_cfg(getattr(explicit_cfg, model_name), obs, obs_groups,
                                             obs_set, output_dim)
        torch.manual_seed(123)
        legacy_model = _rnn_model_from_cfg(getattr(legacy_cfg, model_name), obs, obs_groups,
                                           obs_set, output_dim)

        explicit_state = explicit_model.state_dict()
        legacy_state = legacy_model.state_dict()
        assert explicit_state.keys() == legacy_state.keys()
        for key in explicit_state:
            assert explicit_state[key].shape == legacy_state[key].shape
            torch.testing.assert_close(explicit_state[key], legacy_state[key])


def test_documented_train_commands_have_enough_environments_for_minibatches() -> None:
    """Documented trainer commands must not create empty recurrent PPO minibatches."""
    num_mini_batches = Rizon4sGravDisplayportInsertionRNNPPORunnerCfg().algorithm.num_mini_batches
    commands_with_num_envs = 0
    offenders: list[str] = []

    for path in _DOCUMENTATION_PATHS:
        for line_number, command in _documented_train_commands(path):
            match = _NUM_ENVS.search(command)
            if match is None:
                continue
            commands_with_num_envs += 1
            num_envs = int(match.group(1))
            if num_envs < num_mini_batches:
                offenders.append(
                    f"{path}:{line_number} uses --num_envs {num_envs}: {command}")

    assert commands_with_num_envs > 0
    assert offenders == [], (
        "Documented recurrent PPO commands create empty minibatches:\n" + "\n".join(offenders)
    )
