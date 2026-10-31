# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""LEAPP export contracts for trained Isaac Lab policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

TASK_SPACE_EXPORT_MODEL_NAME = "DisplayPortTaskSpace"

_ROT6D_ELEMENTS = ["r00", "r01", "r02", "r10", "r11", "r12"]
_TASK_SPACE_INPUT_SPEC = (
    ("eef_pos", slice(0, 3), ["x", "y", "z"], "eef_pose_pos", "state/body/position", 0),
    ("eef_rot_6d", slice(3, 9), _ROT6D_ELEMENTS, "eef_pose_rot6d", "state/body/rotation_6d", 3),
    ("socket_kp_pos", slice(9, 12), ["x", "y",
                                     "z"], "socket_kp_pose_pos", "state/body/position", 9),
    (
        "socket_kp_rot_6d",
        slice(12, 18),
        _ROT6D_ELEMENTS,
        "socket_kp_pose_rot6d",
        "state/body/rotation_6d",
        12,
    ),
)
_TASK_SPACE_INPUT_TERM_METADATA = {
    "eef_pos": ("eef_pos", 3, ["x", "y", "z"], "eef_pose_pos", "state/body/position"),
    "eef_rot_6d": ("eef_rot_6d", 6, _ROT6D_ELEMENTS, "eef_pose_rot6d", "state/body/rotation_6d"),
    "socket_kp_pos": ("socket_kp_pos", 3, ["x", "y",
                                           "z"], "socket_kp_pose_pos", "state/body/position"),
    "socket_kp_rot_6d": ("socket_kp_rot_6d", 6, _ROT6D_ELEMENTS, "socket_kp_pose_rot6d",
                         "state/body/rotation_6d"),
    "socket_pos": ("socket_kp_pos", 3, ["x", "y",
                                        "z"], "socket_kp_pose_pos", "state/body/position"),
    "tool_pos": ("eef_pos", 3, ["x", "y", "z"], "eef_pose_pos", "state/body/position"),
    "tool_rot_6d": ("eef_rot_6d", 6, _ROT6D_ELEMENTS, "eef_pose_rot6d", "state/body/rotation_6d"),
    "socket_rot_6d": ("socket_kp_rot_6d", 6, _ROT6D_ELEMENTS, "socket_kp_pose_rot6d",
                      "state/body/rotation_6d"),
}
_ACTION_ELEMENT_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_axis_angle_x",
    "delta_axis_angle_y",
    "delta_axis_angle_z",
]


class ExportContract(Protocol):
    """Task-specific policy I/O annotation contract."""

    name: str
    graph_name: str
    skip_automatic_env_patch: bool

    def prepare_observations(self, graph_name: str, obs, env_cfg, dtype: torch.dtype):
        """Return observations to feed the policy during LEAPP tracing."""

    def export_action(
        self,
        graph_name: str,
        actions,
        env_cfg,
        device,
        dtype: torch.dtype,
        export_method: str,
        clip_actions: float | None,
    ) -> None:
        """Annotate task-specific policy output tensors."""


def task_space_policy_obs(obs):
    """Return the 18D actor observation tensor from an RSL-RL TensorDict."""
    if "policy" not in obs:
        raise KeyError(f"Expected a 'policy' observation group, got keys: {list(obs.keys())}")
    policy_obs = obs["policy"]
    if policy_obs.shape[-1] != 18:
        raise ValueError(
            f"Expected 18D task-space actor observation, got shape {tuple(policy_obs.shape)}")
    return policy_obs


def resolve_task_space_input_spec(env_cfg):
    """Resolve public input slices and actor offsets from environment ABI metadata."""
    obs_order = getattr(env_cfg, "task_space_obs_order", None)
    if obs_order is None:
        return _TASK_SPACE_INPUT_SPEC
    if isinstance(obs_order, str | bytes) or not isinstance(obs_order, Sequence):
        raise TypeError(
            "env_cfg.task_space_obs_order must be a sequence of observation-term names.")

    input_specs_by_name = {}
    seen_terms = set()
    seen_input_names = set()
    start = 0
    for term_name in obs_order:
        if not isinstance(term_name, str):
            raise TypeError("env_cfg.task_space_obs_order entries must be strings.")
        if term_name in seen_terms:
            raise ValueError(f"Duplicate task-space observation term: {term_name!r}.")
        try:
            input_name, width, element_names, source, kind = _TASK_SPACE_INPUT_TERM_METADATA[
                term_name]
        except KeyError as exc:
            known_terms = ", ".join(sorted(_TASK_SPACE_INPUT_TERM_METADATA))
            raise ValueError(
                f"Unknown task-space observation term {term_name!r}. Known terms: {known_terms}."
            ) from exc
        if input_name in seen_input_names:
            raise ValueError(
                f"Duplicate Deploy input resolved from task-space observation metadata: "
                f"{input_name!r}."
            )

        stop = start + width
        input_specs_by_name[input_name] = (input_name, slice(start, stop), element_names, source,
                                           kind, start)
        seen_terms.add(term_name)
        seen_input_names.add(input_name)
        start = stop

    if start != 18:
        raise ValueError(
            f"env_cfg.task_space_obs_order must describe exactly 18 values; resolved {start} "
            f"from {list(obs_order)!r}."
        )
    public_input_names = [entry[0] for entry in _TASK_SPACE_INPUT_SPEC]
    if seen_input_names != set(public_input_names):
        raise ValueError(
            "env_cfg.task_space_obs_order must resolve to exactly the four canonical "
            "DisplayPort Deploy inputs."
        )
    return tuple(input_specs_by_name[name] for name in public_input_names)


def resolve_task_space_input_terms(env_cfg) -> dict[str, str]:
    """Map canonical deploy inputs to their environment observation-term names."""
    obs_order = getattr(env_cfg, "task_space_obs_order", None)
    if obs_order is None:
        return {entry[0]: entry[0] for entry in _TASK_SPACE_INPUT_SPEC}

    # Reuse the ABI validation above before exposing term names to runtime tools.
    resolve_task_space_input_spec(env_cfg)
    return {_TASK_SPACE_INPUT_TERM_METADATA[term_name][0]: term_name for term_name in obs_order}


_TASK_SPACE_DEPLOY_METADATA_FIELDS = (
    "task_space_policy_abi",
    "task_space_backend",
    "task_space_observation_reference",
    "task_space_observation_body_name",
    "task_space_observation_body_offset",
    "task_space_action_reference",
    "task_space_action_body_name",
    "task_space_action_body_offset",
)


def resolve_task_space_deploy_metadata(env_cfg) -> tuple[dict[str, dict], dict]:
    """Resolve additive frame and ABI metadata for task-space deployment bundles."""
    present = [hasattr(env_cfg, field) for field in _TASK_SPACE_DEPLOY_METADATA_FIELDS]
    if not any(present):
        return {}, {}
    if not all(present):
        missing = [
            field
            for field, is_present in zip(_TASK_SPACE_DEPLOY_METADATA_FIELDS, present, strict=True)
            if not is_present
        ]
        raise ValueError(f"Incomplete task-space deployment metadata; missing {missing!r}.")

    policy_abi = str(env_cfg.task_space_policy_abi)
    backend = str(env_cfg.task_space_backend)
    if not policy_abi or not backend:
        raise ValueError("Task-space policy ABI and backend metadata must be non-empty strings.")

    def offset(name: str) -> list[float]:
        values = torch.as_tensor(getattr(env_cfg, name), dtype=torch.float64).flatten()
        if values.numel() != 3 or not bool(torch.isfinite(values).all()):
            raise ValueError(f"{name} must contain three finite values.")
        return [float(value) for value in values.tolist()]

    common = {"policy_abi": policy_abi, "physics_backend": backend}
    observation = {
        **common,
        "coordinate_frame": "world",
        "pose_reference": str(env_cfg.task_space_observation_reference),
        "body_name": str(env_cfg.task_space_observation_body_name),
        "body_offset": offset("task_space_observation_body_offset"),
    }
    action = {
        **common,
        "pose_reference": str(env_cfg.task_space_action_reference),
        "body_name": str(env_cfg.task_space_action_body_name),
        "body_offset": offset("task_space_action_body_offset"),
    }
    return {"eef_pos": observation, "eef_rot_6d": observation}, action


def split_and_annotate_task_space_obs(graph_name: str,
                                      policy_obs,
                                      input_spec=_TASK_SPACE_INPUT_SPEC,
                                      input_metadata: dict[str, dict] | None = None):
    """Expose EEF-first inputs while reconstructing the checkpoint trained actor order."""
    input_metadata = input_metadata or {}
    import leapp
    from leapp.utils.tensor_description import TensorSemantics

    actor_parts = []
    for name, index, element_names, source, kind, actor_offset in input_spec:
        part = leapp.annotate.input_tensors(
            graph_name,
            TensorSemantics(
                name=name,
                ref=policy_obs[:, index],
                kind=kind,
                element_names=[element_names],
                extra={
                    "source": source,
                    **input_metadata.get(name, {})
                },
            ),
        )
        actor_parts.append((actor_offset, part))
    actor_parts.sort(key=lambda item: item[0])
    return torch.cat([part for _, part in actor_parts], dim=-1)


def export_task_space_action(graph_name: str,
                             tensor,
                             export_method: str,
                             extra_metadata: dict | None = None) -> None:
    """Annotate the deploy-facing clipped and scaled task-space action."""
    metadata = {
        "isaaclab_connection": "action:arm_action:pose_rel",
        "target_types": ["pose_rel"],
    }
    metadata.update(extra_metadata or {})
    import leapp
    from leapp.utils.tensor_description import TensorSemantics

    leapp.annotate.output_tensors(
        graph_name,
        TensorSemantics(
            name="arm_action",
            ref=tensor,
            kind="target/body/pose_relative",
            element_names=[_ACTION_ELEMENT_NAMES],
            extra=metadata,
        ),
        export_with=export_method,
    )


def task_space_action_scale(env_cfg, device, dtype):
    """Return configured position and orientation scales expanded over three axes."""
    action_cfg = env_cfg.actions.arm_action
    for attr in ("position_scale", "orientation_scale"):
        if not hasattr(action_cfg, attr):
            raise AttributeError(
                f"Task-space export requires an operational-space action exposing {attr!r}, "
                f"got {type(action_cfg).__name__}.")

    def expand_scale(value, name):
        scale = torch.as_tensor(value, device=device, dtype=dtype).flatten()
        if scale.numel() == 1:
            scale = scale.repeat(3)
        elif scale.numel() != 3:
            raise ValueError(
                f"{name} must contain either one or three values; got {scale.numel()}.")
        if not bool(torch.isfinite(scale).all()):
            raise ValueError(f"{name} must contain only finite values.")
        return scale

    position_scale = expand_scale(action_cfg.position_scale, "position_scale")
    orientation_scale = expand_scale(action_cfg.orientation_scale, "orientation_scale")
    return torch.cat((position_scale, orientation_scale)).unsqueeze(0)


def process_task_space_action(actions, scale, clip_actions: float | None):
    """Apply the RSL-RL clip contract before the OSC pose-delta scale."""
    if clip_actions is not None:
        clip_actions = float(clip_actions)
        actions = actions.clamp(-clip_actions, clip_actions)
    return actions * scale


def recover_task_space_policy_action(processed_actions, scale, clip_actions: float | None):
    """Invert the deploy pose-delta transform for Isaac Lab simulation playback."""
    if bool((scale == 0).any()):
        raise ValueError("Task-space action scale must be non-zero for simulation playback.")
    actions = processed_actions / scale
    if clip_actions is not None:
        clip_actions = float(clip_actions)
        actions = actions.clamp(-clip_actions, clip_actions)
    return actions


@dataclass(frozen=True)
class DisplayPortTaskSpaceContract:
    """DisplayPort task-space Deploy contract.

    The policy still receives the byte-identical 18D trained observation, but LEAPP
    exposes that vector as named EEF/socket pose tensors and emits a clipped, scaled
    Cartesian pose delta.
    """

    name: str = "displayport_task_space"
    graph_name: str = TASK_SPACE_EXPORT_MODEL_NAME
    skip_automatic_env_patch: bool = True

    def prepare_observations(self, graph_name: str, obs, env_cfg, dtype: torch.dtype):
        policy_obs = task_space_policy_obs(obs).to(dtype=dtype)
        obs_for_policy = obs.clone()
        input_spec = resolve_task_space_input_spec(env_cfg)
        input_metadata, _ = resolve_task_space_deploy_metadata(env_cfg)
        obs_for_policy["policy"] = split_and_annotate_task_space_obs(graph_name,
                                                                     policy_obs,
                                                                     input_spec=input_spec,
                                                                     input_metadata=input_metadata)
        return obs_for_policy

    def export_action(
        self,
        graph_name: str,
        actions,
        env_cfg,
        device,
        dtype: torch.dtype,
        export_method: str,
        clip_actions: float | None,
    ) -> None:
        from isaaclab_training.mdp.actions import _pose_rel_action_extra

        scale = task_space_action_scale(env_cfg, device, dtype)
        processed_action = process_task_space_action(actions, scale, clip_actions)
        _, action_metadata = resolve_task_space_deploy_metadata(env_cfg)
        action_extra = _pose_rel_action_extra(
            position_scale=env_cfg.actions.arm_action.position_scale,
            orientation_scale=env_cfg.actions.arm_action.orientation_scale,
            target_types=env_cfg.actions.arm_action.controller_cfg.target_types,
        )
        action_extra.update(action_metadata)
        action_extra["clip_actions"] = clip_actions
        export_task_space_action(graph_name,
                                 processed_action,
                                 export_method,
                                 extra_metadata=action_extra)


CONTRACTS: dict[str, ExportContract] = {
    "displayport_task_space": DisplayPortTaskSpaceContract(),
}


def get_export_contract(name: str | None) -> ExportContract | None:
    """Resolve an export contract by name."""
    if name in (None, "", "none"):
        return None
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        available = ", ".join(sorted(CONTRACTS))
        raise ValueError(
            f"Unknown export contract '{name}'. Available contracts: {available}") from exc
