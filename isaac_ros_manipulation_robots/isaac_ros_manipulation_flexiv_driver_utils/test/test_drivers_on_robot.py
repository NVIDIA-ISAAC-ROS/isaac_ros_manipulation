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
Launch test for Flexiv driver bringup using flexiv_bringup rizon.launch.py.

Runs only when ENABLE_MANIPULATOR_TESTING=on_robot (real hardware, FLEXIV_ROBOT_SN).
Otherwise launches a no-op static transform so CI passes without flexiv_bringup.
"""

import os
import unittest

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import launch_testing

import pytest
import rclpy
from rclpy.node import Node as RclpyNode

from sensor_msgs.msg import JointState


RUN_TEST = (
    os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
)
ROBOT_SN = os.environ.get('FLEXIV_ROBOT_SN', 'Rizon4s-062839')


@pytest.mark.rostest
def generate_test_description():
    """Launch Flexiv drivers and verify they come up correctly."""
    test_nodes = []
    node_startup_delay = 1.0

    if RUN_TEST:
        flexiv_bringup_launch_dir = os.path.join(
            get_package_share_directory('flexiv_bringup'), 'launch'
        )
        node_startup_delay = 12.0
        launch_args = {
            'robot_sn': ROBOT_SN,
            'rizon_type': 'Rizon4s',
            'use_fake_hardware': 'false',
            'start_rviz': 'false',
            'robot_controller': 'rizon_arm_controller',
        }
        test_nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(flexiv_bringup_launch_dir, 'rizon.launch.py')
                ),
                launch_arguments=launch_args.items(),
            )
        )
    else:
        test_nodes.append(
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_transform_publisher',
                output='screen',
                arguments=[
                    '--x', '0', '--y', '0', '--z', '0',
                    '--yaw', '0', '--pitch', '0', '--roll', '0',
                    '--frame-id', 'world', '--child-frame-id', 'base_link',
                ],
            )
        )

    global_configs = []
    if RUN_TEST:
        global_configs.append(
            SetLaunchConfiguration('robot_controller', 'rizon_arm_controller')
        )

    return LaunchDescription(
        global_configs
        + test_nodes
        + [
            TimerAction(
                period=node_startup_delay,
                actions=[launch_testing.actions.ReadyToTest()],
            ),
        ]
    )


class TestFlexivDriverBringup(unittest.TestCase):
    """Verify Flexiv driver nodes come up and /joint_states is published."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = RclpyNode('flexiv_driver_bringup_test_node')
        cls.received_msgs = []
        cls.sub = cls.node.create_subscription(
            JointState,
            '/joint_states',
            lambda msg: cls.received_msgs.append(msg),
            10,
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_driver_nodes_running(self):
        """Verify expected driver nodes are running."""
        if not RUN_TEST:
            self.skipTest(
                'Set ENABLE_MANIPULATOR_TESTING=on_robot to run this test.'
            )
        timeout_sec = 30.0
        end_time = self.node.get_clock().now().nanoseconds + int(
            timeout_sec * 1e9
        )

        running_nodes = []
        while self.node.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self.node, timeout_sec=1.0)
            running_nodes = self.node.get_node_names()
            if 'rizon_arm_controller' in running_nodes:
                break

        expected_nodes = [
            'controller_manager',
            'joint_state_broadcaster',
            'rizon_arm_controller',
            'robot_state_publisher',
            'joint_state_publisher',
        ]

        for expected in expected_nodes:
            self.assertIn(
                expected,
                running_nodes,
                f'{expected} is not running. Active nodes: {running_nodes}',
            )

    def test_joint_states_published(self):
        """Verify /joint_states receives messages with 7 arm joints."""
        if not RUN_TEST:
            self.skipTest(
                'Set ENABLE_MANIPULATOR_TESTING=on_robot to run this test.'
            )
        timeout_sec = 30.0
        end_time = self.node.get_clock().now().nanoseconds + int(
            timeout_sec * 1e9
        )

        while (
            self.node.get_clock().now().nanoseconds < end_time
            and len(self.received_msgs) == 0
        ):
            rclpy.spin_once(self.node, timeout_sec=0.5)

        self.assertGreater(
            len(self.received_msgs),
            0,
            '/joint_states topic did not receive any messages '
            f'within {timeout_sec}s.',
        )

        msg = self.received_msgs[0]
        self.assertGreaterEqual(
            len(msg.name),
            7,
            f'Expected at least 7 joint names, got {len(msg.name)}: '
            f'{msg.name}',
        )
