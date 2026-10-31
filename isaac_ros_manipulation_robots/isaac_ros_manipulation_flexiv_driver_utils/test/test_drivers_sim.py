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

"""Sim driver launch test; runs only when ENABLE_MANIPULATOR_TESTING=on_robot."""

import os
import unittest

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils import load_yaml_params

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node as RosNode
import launch_testing

import pytest
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


RUN_TEST = (
    os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
)


@pytest.mark.rostest
def generate_test_description():
    """Launch Flexiv drivers and verify commands are published."""
    test_actions = []
    ready_delay_sec = 1.0

    if RUN_TEST:
        ready_delay_sec = 5.0
        flexiv_driver_launch_dir = os.path.join(
            get_package_share_directory(
                'isaac_ros_manipulation_flexiv_driver_utils'),
            'launch')
        test_yaml_config = os.path.join(
            get_package_share_directory(
                'isaac_ros_manipulation_flexiv_driver_utils'),
            'params',
            'flexiv_rizon_grav.yaml'
        )
        # Local seed params: the bringup simulation params reference that package
        # (not installed under minimal Jenkins/colcon selections). See params/
        # flexiv_driver_sim_test_seed_params.yaml header.
        sim_seed_params_config = os.path.join(
            get_package_share_directory(
                'isaac_ros_manipulation_flexiv_driver_utils'),
            'params',
            'flexiv_driver_sim_test_seed_params.yaml',
        )
        params = load_yaml_params(sim_seed_params_config)
        params.update(load_yaml_params(test_yaml_config))
        params.update({
            'headless': 'true',
            'workflow_type': 'POSE_TO_POSE',
            'rizon_type': 'Rizon4s',
        })

        test_actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    flexiv_driver_launch_dir,
                    '/flexiv_rizon_sim_driver.launch.py']),
                launch_arguments={
                    key: str(value) for key, value
                    in params.items()
                }.items()))
    else:
        test_actions.append(
            RosNode(
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

    return LaunchDescription([
        TimerAction(period=0.0, actions=test_actions),
        TimerAction(
            period=ready_delay_sec,
            actions=[launch_testing.actions.ReadyToTest()]),
    ])


class TestFlexivDriverCommands(unittest.TestCase):
    """Verify that /rizon_sim_command is being published by ros2_control."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node('flexiv_driver_test_node')
        cls.received_msgs = []
        cls.sub = cls.node.create_subscription(
            JointState,
            '/rizon_sim_command',
            lambda msg: cls.received_msgs.append(msg),
            10
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_joint_commands_published(self):
        """Check that joint commands are published within timeout."""
        if not RUN_TEST:
            self.skipTest(
                'Set ENABLE_MANIPULATOR_TESTING=on_robot to run this test.'
            )
        timeout_sec = 15.0
        end_time = self.node.get_clock().now().nanoseconds + int(
            timeout_sec * 1e9)

        while (self.node.get_clock().now().nanoseconds < end_time
               and len(self.received_msgs) == 0):
            rclpy.spin_once(self.node, timeout_sec=0.5)

        self.assertGreater(
            len(self.received_msgs), 0,
            '/rizon_sim_command topic did not receive any messages '
            f'within {timeout_sec}s. TopicBasedSystem ros2_control plugin '
            'should be publishing initial joint positions.')

        msg = self.received_msgs[0]
        self.assertEqual(
            len(msg.name), 7,
            f'Expected 7 joint names, got {len(msg.name)}: {msg.name}')
        expected_joints = {
            'joint1', 'joint2', 'joint3', 'joint4',
            'joint5', 'joint6', 'joint7'}
        self.assertEqual(
            set(msg.name), expected_joints,
            f'Joint names mismatch: {msg.name}')
