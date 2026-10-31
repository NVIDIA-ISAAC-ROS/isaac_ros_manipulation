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

import importlib

import gymnasium as gym
import isaaclab_training.tasks  # noqa: F401

EXPECTED_TASKS = {
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-NoJointVel-ROS-Inference",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-Joint-ROS-Inference",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-Newton-ROS-Inference",
    "IsaacTraining-DisplayPortInsertion-Rizon4s-TaskSpace-ROS-Inference",
}

EXPECTED_ALIASES = {
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-ROS-Inference",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-Newton-ROS-Inference",
    "IsaacContrib-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-ROS-Inference",
}


def test_displayport_tasks_register():
    task_ids = {
        spec.id
        for spec in gym.registry.values()
        if spec.id.startswith("IsaacTraining-DisplayPortInsertion")
    }
    assert task_ids == EXPECTED_TASKS


def test_legacy_displayport_aliases_register():
    aliases = {
        spec.id
        for spec in gym.registry.values() if spec.id.startswith("IsaacContrib-Deploy-Displayport")
    }
    assert aliases == EXPECTED_ALIASES


def test_task_entry_points_are_package_qualified():
    for task_id in EXPECTED_TASKS:
        spec = gym.spec(task_id)
        assert spec.entry_point == (
            "isaaclab_training.tasks.displayport_insertion.insertion_env:DisplayportInsertionEnv")
        assert spec.kwargs["env_cfg_entry_point"].startswith(
            "isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s.")
        assert spec.kwargs["rsl_rl_cfg_entry_point"].startswith(
            "isaaclab_training.tasks.displayport_insertion.config.displayport_rizon_4s.agents.")
        assert spec.kwargs["default_agent"] == "rsl_rl"


def test_config_entry_points_resolve():
    for task_id in EXPECTED_TASKS:
        spec = gym.spec(task_id)
        for key, value in spec.kwargs.items():
            if not key.endswith("_entry_point") or not isinstance(value, str) or ":" not in value:
                continue
            module_name, attribute_name = value.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            assert hasattr(module, attribute_name), f"{task_id}: {key}={value}"
