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

"""Flexiv Rizon 4s DisplayPort insertion task registrations.

Separate ``-Play`` task ids are intentionally not registered. Isaac Lab retired them in favour
of :meth:`play_mode` overrides that the play scripts apply to the training configuration, so
playing a training id picks up the task's play-mode behaviour automatically. Any remaining
``-Play`` id resolves to its training id with a deprecation warning.
"""

from __future__ import annotations

import gymnasium as gym

from . import agents

_INSERTION_ENV_ENTRY = (
    "isaaclab_training.tasks.displayport_insertion.insertion_env:DisplayportInsertionEnv"
)
_AGENT = f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionRNNPPORunnerCfg"
# Task-space tasks use 128-step rollouts (same sample budget, see rsl_rl_ppo_cfg.py).
_AGENT_TASK_SPACE = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionTaskSpaceRNNPPORunnerCfg"
)
_AGENT_NEWTON = (
    f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGravDisplayportInsertionNewtonRNNPPORunnerCfg"
)


def _register(task_id: str, env_cfg_entry_point: str) -> None:
    """Register one Rizon4s DisplayPort task."""
    if ".task_space_newton_" in env_cfg_entry_point:
        agent = _AGENT_NEWTON
    elif ".task_space_" in env_cfg_entry_point:
        agent = _AGENT_TASK_SPACE
    else:
        agent = _AGENT
    gym.register(
        id=task_id,
        entry_point=_INSERTION_ENV_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": agent,
            "default_agent": "rsl_rl",
        },
    )


_TASKS = {
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint":
    (f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionEnvCfg"),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel":
    (f"{__name__}.joint_pos_env_cfg:Rizon4sGravDisplayportInsertionNoJointVelEnvCfg"),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference":
    (f"{__name__}.ros_inference_env_cfg:"
     f"Rizon4sGravDisplayportInsertionNoJointVelROSInferenceEnvCfg"
     ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference":
    (f"{__name__}.ros_inference_env_cfg:Rizon4sGravDisplayportInsertionROSInferenceEnvCfg"),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace":
    (f"{__name__}.task_space_env_cfg:Rizon4sTaskSpaceDisplayportInsertionEnvCfg"),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference":
    (f"{__name__}.task_space_ros_inference_env_cfg:"
     f"Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg"
     ),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton":
    (f"{__name__}.task_space_newton_env_cfg:"
     f"Rizon4sTaskSpaceNewtonDisplayportInsertionEnvCfg"),
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference":
    (f"{__name__}.task_space_newton_ros_inference_env_cfg:"
     f"Rizon4sTaskSpaceNewtonDisplayportInsertionROSInferenceEnvCfg"),
}

_ALIASES = {
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-Joint"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton"),
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference":
    ("IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference"),
}

for _task_id, _cfg in _TASKS.items():
    _register(_task_id, _cfg)

for _alias, _target in _ALIASES.items():
    _register(_alias, _TASKS[_target])
