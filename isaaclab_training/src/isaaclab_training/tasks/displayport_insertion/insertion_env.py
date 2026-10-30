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

"""Manager-based insertion env with task-success detection and logging.

Adds success metrics to ``extras["log"]`` for RSL-RL without changing the MDP
observation, action, reward, or termination logic.

.. note::
   Manager-based environments are normally fully configuration driven, with no environment
   subclass. This class is a deliberate exception, kept only to publish per-step success
   metrics, because no manager currently offers a per-step hook that can write arbitrary
   ``extras["log"]`` entries:

   - ``EventManager`` supports only ``startup``, ``reset`` and ``interval`` modes.
   - ``RewardManager`` logs episodic sums under ``Episode_Reward/<term>``; a zero-weight term
     logs zero, so it cannot carry a metric.
   - ``CurriculumManager`` and ``TerminationManager`` run at reset or termination only, so
     neither can express a per-step rate.

   Expressing these through managers would rename the keys and change their semantics from
   per-step to episodic, which would break comparison against existing training runs. The MDP
   itself stays fully cfg driven; this class adds logging only, and overrides no MDP behaviour.
   If a per-step metrics hook lands in Isaac Lab, this class should be removed in favour of it.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms
import torch
import warp as wp


def _keypoint_offsets_6d(device: torch.device) -> torch.Tensor:
    """Return the 7 unit keypoint offsets used by the keypoint reward."""
    corners = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                           device=device,
                           dtype=torch.float32)
    return torch.cat((corners, -corners[-3:]), dim=0)


_MIN_TRIANGLE_PAIRS = 2**20


def _scale_newton_triangle_pairs(cfg) -> None:
    """Size Newton's scene-wide triangle-pair buffer from the final environment count.

    ``max_triangle_pairs`` is one buffer shared by all environments, while ``scene.num_envs`` is
    usually overridden after the config is built. Configs that set
    ``newton_triangle_pairs_per_env`` get ``per_env * num_envs`` (at least 2^20) here, before the
    simulation is created.
    """
    per_env = getattr(cfg, "newton_triangle_pairs_per_env", None)
    collision_cfg = getattr(cfg.sim.physics, "collision_cfg", None)
    if per_env is None or collision_cfg is None:
        return
    collision_cfg.max_triangle_pairs = max(_MIN_TRIANGLE_PAIRS,
                                           int(per_env) * cfg.scene.num_envs)


class DisplayportInsertionEnv(ManagerBasedRLEnv):
    """Manager-based RL env that logs insertion success metrics during training.

    The following scalars are added to ``extras["log"]``:

    - ``Metrics/success_rate``: fraction of envs within the success threshold
    - ``Metrics/plug_socket_pos_error_m``: mean mate-point distance (m)
    - ``Metrics/plug_socket_keypoint_dist_m``: mean keypoint distance (m)
    - ``Metrics/terminal_success_rate``: success fraction at episode reset
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        _scale_newton_triangle_pairs(cfg)
        # Size the PhysX collision stack to the final env count (``--num_envs`` is applied after
        # the
        # cfg is built). Buffer capacity only: contacts are identical as long as PhysX does not
        # overflow.
        per_1024 = getattr(cfg, "physx_collision_stack_bytes_per_1024_envs", None)
        if per_1024 is not None and hasattr(cfg.sim.physics, "gpu_collision_stack_size"):
            stack = max(int(cfg.physx_collision_stack_min_bytes),
                        -(-int(per_1024) * cfg.scene.num_envs // 1024))
            cfg.sim.physics.gpu_collision_stack_size = -(-stack //
                                                         2**20) * 2**20  # round up to whole MiB
        per_env_patches = getattr(cfg, "physx_rigid_patches_per_env", None)
        if per_env_patches is not None and hasattr(cfg.sim.physics, "gpu_max_rigid_patch_count"):
            cfg.sim.physics.gpu_max_rigid_patch_count = max(
                int(cfg.sim.physics.gpu_max_rigid_patch_count),
                int(per_env_patches) * cfg.scene.num_envs)
        per_env_contacts = getattr(cfg, "physx_rigid_contacts_per_env", None)
        if per_env_contacts is not None and hasattr(
                cfg.sim.physics, "gpu_max_rigid_contact_count"):
            contacts = int(cfg.sim.physics.gpu_max_rigid_contact_count)
            while contacts < int(per_env_contacts) * cfg.scene.num_envs:
                contacts *= 2
            cfg.sim.physics.gpu_max_rigid_contact_count = contacts
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._log_success_metrics: bool = bool(getattr(cfg, "log_success_metrics", True))
        self._success_socket_asset: str = getattr(cfg, "success_socket_asset", "dp_socket")
        self._success_plug_asset: str = getattr(cfg, "success_plug_asset", "dp_plug")
        self._success_pos_threshold: float = float(getattr(cfg, "success_pos_threshold", 0.003))
        self._success_keypoint_scale: float = float(getattr(cfg, "success_keypoint_scale", 0.15))

        device = self.device
        self._success_socket_offset = torch.tensor(getattr(cfg, "success_socket_offset",
                                                           [0.0, 0.0, 0.0]),
                                                   device=device,
                                                   dtype=torch.float32)
        self._success_plug_offset = torch.tensor(getattr(cfg, "success_plug_offset",
                                                         [0.0, 0.0, 0.0]),
                                                 device=device,
                                                 dtype=torch.float32)
        self._success_plug_goal_rot_inv = torch.tensor(getattr(cfg, "success_plug_goal_rot_inv",
                                                               [0.0, 0.0, 0.0, 1.0]),
                                                       device=device,
                                                       dtype=torch.float32)

        self._success_identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                                                   device=device,
                                                   dtype=torch.float32).repeat(self.num_envs, 1)
        self._success_kp_offsets = _keypoint_offsets_6d(device) * self._success_keypoint_scale

    def _compute_success(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-env success mask, mate-point distance, and keypoint distance."""
        socket = self.scene[self._success_socket_asset]
        plug = self.scene[self._success_plug_asset]

        socket_pos = wp.to_torch(socket.data.root_pos_w)
        socket_quat = wp.to_torch(socket.data.root_quat_w)
        plug_pos = wp.to_torch(plug.data.root_pos_w)
        plug_quat = wp.to_torch(plug.data.root_quat_w)

        n = self.num_envs
        socket_off = self._success_socket_offset.unsqueeze(0).expand(n, -1)
        plug_off = self._success_plug_offset.unsqueeze(0).expand(n, -1)
        plug_goal_rot_inv = self._success_plug_goal_rot_inv.unsqueeze(0).expand(n, -1)

        # Mate reference frames (same construction as keypoint_two_body_error)
        kp_pos_s, kp_quat_s = combine_frame_transforms(socket_pos, socket_quat, socket_off,
                                                       self._success_identity_quat)
        kp_pos_p, kp_quat_p = combine_frame_transforms(plug_pos, plug_quat, plug_off,
                                                       plug_goal_rot_inv)

        pos_error = torch.linalg.norm(kp_pos_p - kp_pos_s, dim=-1)

        k = self._success_kp_offsets.shape[0]
        offs_flat = self._success_kp_offsets.unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)
        ident_flat = self._success_identity_quat.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4)

        kp_s = combine_frame_transforms(
            kp_pos_s.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3),
            kp_quat_s.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            offs_flat,
            ident_flat,
        )[0].reshape(n, k, 3)
        kp_p = combine_frame_transforms(
            kp_pos_p.unsqueeze(1).expand(-1, k, -1).reshape(-1, 3),
            kp_quat_p.unsqueeze(1).expand(-1, k, -1).reshape(-1, 4),
            offs_flat,
            ident_flat,
        )[0].reshape(n, k, 3)
        keypoint_dist = torch.linalg.norm(kp_p - kp_s, dim=-1).mean(dim=-1)

        is_success = pos_error < self._success_pos_threshold
        return is_success, pos_error, keypoint_dist

    def step(self, action: torch.Tensor):
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        if getattr(self, "_log_success_metrics", False):
            is_success, pos_error, keypoint_dist = self._compute_success()
            log = self.extras.setdefault("log", {})
            log["Metrics/success_rate"] = is_success.float().mean()
            log["Metrics/plug_socket_pos_error_m"] = pos_error.mean()
            log["Metrics/plug_socket_keypoint_dist_m"] = keypoint_dist.mean()
        return obs_buf, reward_buf, terminated, time_outs, self.extras

    def _reset_idx(self, env_ids):
        terminal_success = None
        if getattr(self, "_log_success_metrics", False):
            is_success, _, _ = self._compute_success()
            terminal_success = is_success[env_ids].float().mean()

        super()._reset_idx(env_ids)

        if terminal_success is not None:
            self.extras["log"]["Metrics/terminal_success_rate"] = terminal_success
