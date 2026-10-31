#!/usr/bin/env python3
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
"""
Gripper action server and command combiner for Flexiv Grav gripper in Isaac Sim.

Subscribes to arm-only commands from ros2_control on /rizon_arm_command and
republishes them to /rizon_sim_command.  The current finger joint target is
published separately on /rizon_gripper_joint_command so the USD scene can run a
physics-step-aware gripper controller.  The GRAV USD uses mimic joints for the
remaining fingers, so only finger_joint should be commanded directly.

Also exposes a ParallelGripperCommand action server so the behavior tree can
open/close the gripper.
"""
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

COMMAND_PUBLISH_PERIOD_SEC = 0.1
GRIPPER_COMMAND_DURATION_SEC = 3.0
GRIPPER_COMMAND_HOLD_SEC = 1.0
GRIPPER_JOINT_COMMAND_TOPIC = '/rizon_gripper_joint_command'

GRAV_JOINT_NAMES = ['finger_joint']


class IsaacSimGravGripperActionServer(Node):

    def __init__(self):
        super().__init__('isaac_sim_grav_gripper_action_server')

        self._action_server = ActionServer(
            self,
            ParallelGripperCommand,
            '/flexiv_gripper_node/gripper_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self._combined_pub = self.create_publisher(
            JointState, '/rizon_sim_command', 10)

        self._arm_cmd_sub = self.create_subscription(
            JointState, '/rizon_arm_command',
            self._arm_command_callback, 10)

        self._finger_joint_pos = 0.0
        self._finger_joint_max_effort = 0.0
        self._latest_arm_command = None
        self._action_in_progress = False
        self._gripper_joint_pub = self.create_publisher(
            JointState, GRIPPER_JOINT_COMMAND_TOPIC, 10)
        self.get_logger().info(
            f'Publishing GRAV finger_joint commands on {GRIPPER_JOINT_COMMAND_TOPIC}')

    @staticmethod
    def _smoothstep(alpha: float) -> float:
        alpha = max(0.0, min(1.0, alpha))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal request for Grav gripper')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def _arm_command_callback(self, msg: JointState):
        """Combine arm commands with current gripper state and publish."""
        self._latest_arm_command = msg
        self._publish_combined_command(msg)

    def _publish_combined_command(self, msg: JointState):
        """Publish the latest arm command with the current gripper joint targets."""
        combined = JointState()
        combined.header = Header()
        combined.header.stamp = self.get_clock().now().to_msg()
        combined.name = list(msg.name)
        combined.position = list(msg.position)
        combined.velocity = []
        combined.effort = []
        self._combined_pub.publish(combined)
        self._publish_gripper_joint_command()

    def _publish_gripper_joint_command(self):
        """Publish the GRAV finger target for Isaac-side physics-loop control."""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(GRAV_JOINT_NAMES)
        msg.position = self._build_gripper_positions(self._finger_joint_pos)
        msg.velocity = []
        msg.effort = (
            [float(self._finger_joint_max_effort)]
            if self._finger_joint_max_effort
            else []
        )
        self._gripper_joint_pub.publish(msg)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal for Grav gripper')
        self._action_in_progress = True

        cmd = goal_handle.request.command
        desired_position = cmd.position[0] if cmd.position else 0.0
        self._finger_joint_max_effort = cmd.effort[0] if cmd.effort else 0.0
        start_position = float(self._finger_joint_pos)
        publish_steps = max(
            1,
            int(round(
                (GRIPPER_COMMAND_DURATION_SEC + GRIPPER_COMMAND_HOLD_SEC) /
                COMMAND_PUBLISH_PERIOD_SEC)),
        )

        result = ParallelGripperCommand.Result()

        for step in range(publish_steps):
            elapsed = step * COMMAND_PUBLISH_PERIOD_SEC
            if GRIPPER_COMMAND_DURATION_SEC > 0.0:
                alpha = self._smoothstep(elapsed / GRIPPER_COMMAND_DURATION_SEC)
            else:
                alpha = 1.0
            self._finger_joint_pos = (
                start_position + alpha * (desired_position - start_position)
            )
            if self._latest_arm_command is not None:
                self._publish_combined_command(self._latest_arm_command)
            else:
                self._publish_gripper_joint_command()

            if goal_handle.status in (
                GoalStatus.STATUS_CANCELING,
                GoalStatus.STATUS_CANCELED,
            ):
                self.get_logger().info('Gripper goal cancelled')

                result.reached_goal = False
                goal_handle.canceled(result)
                return result

            time.sleep(COMMAND_PUBLISH_PERIOD_SEC)

        self._finger_joint_pos = desired_position
        if self._latest_arm_command is not None:
            self._publish_combined_command(self._latest_arm_command)
        else:
            self._publish_gripper_joint_command()

        result.state.position = [desired_position]
        result.state.effort = [float(self._finger_joint_max_effort)]
        result.reached_goal = True

        self._action_in_progress = False
        self.get_logger().info('Gripper goal executed')
        goal_handle.succeed(result)
        return result

    def _build_gripper_positions(self, finger_angle: float):
        """Return the single actuated GRAV gripper joint target."""
        return [finger_angle]


def main(args=None):
    rclpy.init(args=args)
    node = IsaacSimGravGripperActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
