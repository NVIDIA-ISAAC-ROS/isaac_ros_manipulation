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

import time
from typing import List

from control_msgs.action import ParallelGripperCommand
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState


TIMEOUT = 10  # seconds

# Arbitrary arm command to publish for testing
ARM_COMMAND_JOINTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Expected arm joint names for Flexiv
ARM_JOINT_NAMES = [
    'joint1', 'joint2', 'joint3', 'joint4',
    'joint5', 'joint6', 'joint7',
]


class IsaacROSGripperDriverTest(IsaacROSBaseTest):
    """Tests for the Flexiv Grav action server and its output joint commands."""

    def setup_clients(self):
        """Create client/subscribers/publisher once and reuse across tests."""
        if hasattr(self, '_clients_ready') and self._clients_ready:
            return

        self._gripper_commands_rcvd: List[JointState] = []
        self._arm_commands_rcvd: List[JointState] = []

        # Subscriber for the gripper command sent to Isaac Sim
        self._gripper_command_sub = self.node.create_subscription(
            JointState,
            '/rizon_gripper_joint_command',
            lambda msg: self._gripper_commands_rcvd.append(msg),
            10,
        )

        # Subscriber for the forwarded arm command sent to Isaac Sim
        self._arm_command_sub = self.node.create_subscription(
            JointState,
            '/rizon_sim_command',
            lambda msg: self._arm_commands_rcvd.append(msg),
            10,
        )

        # Publisher for the ros2_control arm command input
        self._arm_command_pub = self.node.create_publisher(
            JointState, '/rizon_arm_command', 10)

        # Action client for the gripper command
        self._client = ActionClient(
            self.node, ParallelGripperCommand, '/flexiv_gripper_node/gripper_action',
        )

        # Wait for action server
        end = time.time() + TIMEOUT
        while time.time() < end:
            if self._client.wait_for_server(timeout_sec=0.1):
                self._clients_ready = True
                return
            rclpy.spin_once(self.node, timeout_sec=0.01)

        raise RuntimeError('ParallelGripperCommand action server not available')

    def expected_names(self):
        return ['finger_joint']

    def expected_positions(self, p: float):
        # Flexiv commands only the 'finger_joint'
        return [p]

    def wait_for_gripper_commands(self, min_count: int, timeout_sec: float = TIMEOUT):
        end = time.time() + timeout_sec
        while time.time() < end and len(self._gripper_commands_rcvd) < min_count:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return len(self._gripper_commands_rcvd) >= min_count

    def wait_for_arm_commands(self, min_count: int, timeout_sec: float = TIMEOUT):
        end = time.time() + timeout_sec
        while time.time() < end and len(self._arm_commands_rcvd) < min_count:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return len(self._arm_commands_rcvd) >= min_count

    def publish_arm_command(self, count: int = 1, delay_sec: float = 0.05):
        """Publish an arm command and trigger gripper-target replication."""
        for _ in range(count):
            msg = JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = ARM_JOINT_NAMES
            msg.position = ARM_COMMAND_JOINTS
            self._arm_command_pub.publish(msg)
            time.sleep(delay_sec)

    def send_goal(self, p: float, effort: float = 0.0):
        goal = ParallelGripperCommand.Goal()
        goal.command.position = [p]
        goal.command.effort = [effort]
        return self._client.send_goal_async(goal)

    def wait_for_result(self, send_future):
        """Wait for Flexiv's smooth command trajectory to finish."""
        rclpy.spin_until_future_complete(
            self.node, send_future, timeout_sec=TIMEOUT)
        assert send_future.done(), 'Gripper goal was not accepted before timeout'

        goal_handle = send_future.result()
        assert goal_handle is not None, 'Gripper goal returned no handle'
        assert goal_handle.accepted, 'Gripper goal was rejected'

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self.node, result_future, timeout_sec=TIMEOUT)
        assert result_future.done(), 'Gripper goal did not finish before timeout'
        return result_future.result().result

    def assert_latest_gripper_command(self, p: float, effort: float = 0.0):
        assert self._gripper_commands_rcvd, 'No gripper command received'
        msg = self._gripper_commands_rcvd[-1]
        assert list(msg.name) == self.expected_names()
        exp_pos = self.expected_positions(p)
        assert len(msg.position) == len(exp_pos)
        self.node.get_logger().info(f'msg.position: {msg.position}')
        self.node.get_logger().info(f'exp_pos: {exp_pos}')
        for actual, expected in zip(msg.position, exp_pos):
            assert abs(actual - expected) < 1e-6
        assert list(msg.velocity) == []
        expected_effort = [effort] if effort else []
        assert list(msg.effort) == expected_effort

    def assert_latest_arm_command(self):
        assert self._arm_commands_rcvd, 'No arm command received'
        msg = self._arm_commands_rcvd[-1]
        assert list(msg.name) == ARM_JOINT_NAMES
        assert list(msg.position) == ARM_COMMAND_JOINTS
        assert list(msg.velocity) == []
        assert list(msg.effort) == []

    def test_gripper_command_publishes_expected_joint_command(self):
        """
        Send a command and verify the resulting Grav joint command.

        Checks:
        - Action result reports the requested position and effort
        - Topic publishes the single actuated Grav finger joint
        - Published position and effort match the requested command
        """
        self.setup_clients()
        self._gripper_commands_rcvd.clear()

        p = 0.4
        effort = 5.0
        result = self.wait_for_result(self.send_goal(p, effort))

        assert result.reached_goal
        assert list(result.state.position) == [p]
        assert list(result.state.effort) == [effort]
        assert self.wait_for_gripper_commands(min_count=3, timeout_sec=TIMEOUT), \
            'No gripper command received during goal execution'
        self.assert_latest_gripper_command(p, effort)

    def test_arm_command_replication_and_multiple_goals(self):
        """
        Verify multiple goals and arm-command-triggered target replication.

        Steps:
        - Send goal p1 and validate the completed command
        - Send goal p2 and validate the completed command
        - Publish an arm command and verify it is forwarded to /rizon_sim_command
        - Verify the arm command also triggers replication of the last gripper target
        - Publish additional arm commands and verify the gripper target remains consistent
        """
        self.setup_clients()
        self._gripper_commands_rcvd.clear()

        # First goal
        p1 = 0.1
        result1 = self.wait_for_result(self.send_goal(p1))
        assert result1.reached_goal
        assert self.wait_for_gripper_commands(min_count=1, timeout_sec=TIMEOUT), \
            'No gripper command received for first goal'
        self.assert_latest_gripper_command(p1)

        # Second goal
        self._gripper_commands_rcvd.clear()
        p2 = 0.55
        result2 = self.wait_for_result(self.send_goal(p2))
        assert result2.reached_goal
        assert self.wait_for_gripper_commands(min_count=1, timeout_sec=TIMEOUT), \
            'No gripper command received for second goal'
        self.assert_latest_gripper_command(p2)

        # Clear to focus on republishing after execution via arm commands
        self._gripper_commands_rcvd.clear()
        self._arm_commands_rcvd.clear()

        # Publish a new arm command
        self.publish_arm_command()
        assert self.wait_for_gripper_commands(min_count=1, timeout_sec=TIMEOUT), \
            'No gripper command received on arm-command-triggered republish'
        self.assert_latest_gripper_command(p2)
        assert self.wait_for_arm_commands(
            min_count=1, timeout_sec=TIMEOUT), \
            'No forwarded arm command received'
        self.assert_latest_arm_command()

        # Additional arm commands should keep publishing the last known position
        last_count = len(self._gripper_commands_rcvd)
        self.publish_arm_command(count=2)
        assert self.wait_for_gripper_commands(
            min_count=last_count + 1, timeout_sec=TIMEOUT), \
            'No additional gripper command received on subsequent arm commands'
        self.assert_latest_gripper_command(p2)

    def test_gripper_command_republishes_latest_arm_command(self):
        """
        Verify a gripper goal republishes the latest arm command.

        Steps:
        - Publish an arm command and verify it is forwarded to /rizon_sim_command
        - Send a gripper goal and validate the completed gripper command
        - Verify the cached arm command is republished unchanged during the action
        """
        self.setup_clients()
        self._gripper_commands_rcvd.clear()
        self._arm_commands_rcvd.clear()

        # Seed the driver with an arm command to cache
        self.publish_arm_command()
        assert self.wait_for_arm_commands(
            min_count=1, timeout_sec=TIMEOUT), \
            'No forwarded arm command received before the gripper goal'
        self.assert_latest_arm_command()

        # Clear to focus on commands published by the gripper action
        self._gripper_commands_rcvd.clear()
        self._arm_commands_rcvd.clear()

        p = 0.3
        result = self.wait_for_result(self.send_goal(p))

        assert result.reached_goal
        assert self.wait_for_gripper_commands(min_count=1, timeout_sec=TIMEOUT), \
            'No gripper command received during goal execution'
        self.assert_latest_gripper_command(p)
        assert self.wait_for_arm_commands(
            min_count=1, timeout_sec=TIMEOUT), \
            'Cached arm command was not republished during the gripper goal'
        self.assert_latest_arm_command()


@pytest.mark.rostest
def generate_test_description():
    """Launch the Flexiv Grav gripper driver."""
    gripper_node = Node(
        name='gripper_driver',
        package='isaac_ros_manipulation_flexiv_isaac_sim_utils',
        executable='isaac_sim_grav_gripper_driver.py',
        namespace=IsaacROSGripperDriverTest.generate_namespace(),
    )
    return IsaacROSGripperDriverTest.generate_test_description([gripper_node])
