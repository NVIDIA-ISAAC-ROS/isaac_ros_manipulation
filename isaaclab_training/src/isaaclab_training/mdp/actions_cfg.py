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

"""Deploy-specific action configuration classes."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    OperationalSpaceControllerActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class DeployRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for deploy relative joint actions with explicit LEAPP current-joint input."""

    class_type: type | str = "isaaclab_training.mdp.actions:DeployRelativeJointPositionAction"


@configclass
class DeployOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """OSC action that exports scaled pose deltas for LEAPP instead of joint efforts."""

    class_type: type | str = "isaaclab_training.mdp.actions:DeployOperationalSpaceControllerAction"


@configclass
class DeployDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """DiffIK action that exports scaled pose deltas for LEAPP instead of joint targets."""

    class_type: type | str = (
        "isaaclab_training.mdp.actions:DeployDifferentialInverseKinematicsAction"
    )
