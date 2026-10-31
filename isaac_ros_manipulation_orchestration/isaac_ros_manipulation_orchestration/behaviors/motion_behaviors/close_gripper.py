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

from typing import Dict, Optional

from control_msgs.action import ParallelGripperCommand
from isaac_ros_manipulation_orchestration.behaviors.base_action import BaseActionBehavior
from isaac_ros_manipulation_orchestration.utils.status_types import BehaviorStatus
import py_trees


class CloseGripper(BaseActionBehavior):
    """
    Close gripper using action client.

    This behavior sends a gripper command to close the gripper to a specified
    position with a maximum effort constraint.

    Parameters
    ----------
    name : str
        Name of the behavior
    gripper_action_name : str
        Name of the gripper action server
    close_position : float
        Target position for gripper closure (gripper-specific units)
    close_positions_by_class : Dict[str, float], optional
        Per-object-class closure targets. When the active object's class ID is
        present, its value overrides ``close_position``.
    max_effort : float
        Maximum effort/force to apply during closure (gripper-specific units)

    """

    def __init__(self,
                 name: str,
                 gripper_action_name: str,
                 close_position: float,
                 max_effort: float,
                 close_positions_by_class: Optional[Dict[str, float]] = None
                 ):
        super().__init__(
            name=name,
            action_type=ParallelGripperCommand,
            action_server_name=gripper_action_name
        )

        self.close_position = close_position
        self.close_positions_by_class = close_positions_by_class or {}
        self.max_effort = max_effort

        self.blackboard.register_key(
            key='active_obj_id', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.READ)

    def update(self):
        """
        Drive the gripper closing behavior.

        Returns
        -------
        py_trees.common.Status
            SUCCESS when gripper closes successfully,
            FAILURE when closing fails,
            RUNNING while the action is in progress

        """
        # First, check for server availability and action failures
        status = super().update()
        if status == py_trees.common.Status.FAILURE:
            self.node.get_logger().error('Gripper close failed')
            return py_trees.common.Status.FAILURE

        # Now handle the state machine for this specific behavior
        if self.get_action_state() == BehaviorStatus.IDLE:
            # Start the gripper closing process
            self._trigger_close_gripper()
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.IN_PROGRESS:
            # Wait for the gripper action to complete
            return py_trees.common.Status.RUNNING

        elif self.get_action_state() == BehaviorStatus.SUCCEEDED:
            # Process the gripper close result
            return self._process_result()

        # This should not happen since we're handling all states
        self.node.get_logger().warning(
            f'Unexpected state in {self.name}: {self.get_action_state()}')
        return py_trees.common.Status.FAILURE

    def _trigger_close_gripper(self):
        """Trigger the action call for closing the gripper."""
        gripper_goal = ParallelGripperCommand.Goal()
        close_position, class_id = self._get_close_position()
        gripper_goal.command.position = [close_position]
        gripper_goal.command.effort = [float(self.max_effort)]

        class_suffix = f' for class {class_id}' if class_id is not None else ''
        self.node.get_logger().info(
            f'Closing gripper to position {close_position}{class_suffix}')
        self.send_goal(gripper_goal)

    def _get_close_position(self):
        """Resolve the closure target for the currently active object class."""
        class_id = None
        if (self.blackboard.exists('active_obj_id') and
                self.blackboard.active_obj_id is not None and
                self.blackboard.exists('object_info_cache') and
                self.blackboard.object_info_cache is not None):
            object_info = self.blackboard.object_info_cache.get(
                self.blackboard.active_obj_id, {})
            if object_info.get('class_id') is not None:
                class_id = str(object_info['class_id'])

        close_position = self.close_positions_by_class.get(
            class_id, self.close_position)
        return float(close_position), class_id

    def _process_result(self):
        """
        Process the gripper close action result.

        A close command is considered successful when the action server
        reports either ``reached_goal`` (commanded position reached) or
        ``stalled`` (gripper stopped while applying force, i.e. holding an
        object). Comparing the raw ``position`` field against
        ``close_position`` is not reliable: when closing onto an object the
        gripper legitimately stops short of the commanded width.

        Returns
        -------
        py_trees.common.Status
            SUCCESS if the gripper close completed (either reached the
            commanded position or stalled on an object), FAILURE otherwise.

        """
        result = self.get_action_result()
        reached_goal = getattr(result, 'reached_goal', None)
        stalled = getattr(result, 'stalled', None)
        position = result.state.position[0] if result.state.position else None

        if reached_goal or stalled:
            self.node.get_logger().info(
                f'[{self.name}] Successfully closed gripper '
                f'(reached_goal={reached_goal}, stalled={stalled}, position={position})')
            return py_trees.common.Status.SUCCESS

        self.node.get_logger().error(
            f'[{self.name}] Failed to close gripper to desired position '
            f'(reached_goal={reached_goal}, stalled={stalled}, position={position})')
        return py_trees.common.Status.FAILURE
