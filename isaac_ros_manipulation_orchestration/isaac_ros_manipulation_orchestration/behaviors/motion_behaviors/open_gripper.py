# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from control_msgs.action import ParallelGripperCommand
from isaac_ros_manipulation_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_ros_manipulation_orchestration.utils.status_types import BehaviorStatus
import py_trees


class OpenGripper(BaseActionBehavior):
    """
    Open gripper using action client.

    This behavior sends a gripper command to open the gripper to a specified
    position with a maximum effort constraint.

    Parameters
    ----------
    name : str
        Name of the behavior
    gripper_action_name : str
        Name of the gripper action server
    open_position : float
        Target position for gripper opening (gripper-specific units)
    max_effort : float
        Maximum effort/force to apply during opening (gripper-specific units)

    """

    def __init__(self,
                 name: str,
                 gripper_action_name: str,
                 open_position: float,
                 max_effort: float):
        super().__init__(
            name=name,
            action_type=ParallelGripperCommand,
            action_server_name=gripper_action_name
        )

        self.open_position = open_position
        self.max_effort = max_effort

    def update(self):
        """
        Drive the gripper opening behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when gripper opens successfully,
            FAILURE when opening fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.node.get_logger().error('Gripper open failed')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the gripper opening process
            self._trigger_open_gripper()
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the gripper action to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the gripper open result
            return self._process_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_open_gripper(self):
        """Trigger the action call for opening the gripper."""
        gripper_goal = ParallelGripperCommand.Goal()
        gripper_goal.command.position = [float(self.open_position)]
        gripper_goal.command.effort = [float(self.max_effort)]

        self.node.get_logger().info(f'Opening gripper to position {self.open_position}')
        self.send_goal(gripper_goal)

    def _process_result(self):
        """
        Process the gripper open action result.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the gripper was opened successfully,
            FAILURE otherwise

        """
        # Check if we reached the goal position
        result = self.get_action_result()
        reached_goal = getattr(result, 'reached_goal', None)
        position = result.state.position[0] if result.state.position else None

        if reached_goal:
            self.node.get_logger().info(
                f'[{self.name}] Successfully opened gripper to position: '
                f'{position}')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(
                f'[{self.name}] Failed to open gripper to desired position')
            return py_trees.common.Status.FAILURE
