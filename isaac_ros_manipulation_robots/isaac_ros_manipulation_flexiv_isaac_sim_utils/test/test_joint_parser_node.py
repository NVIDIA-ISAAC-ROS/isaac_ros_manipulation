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

from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import pytest
import rclpy
from sensor_msgs.msg import JointState


TIMEOUT = 10  # seconds


class IsaacROSJointParserTest(IsaacROSBaseTest):
    """This test checks the functionality of the `joint_parser` node."""

    def test_joint_parser_basic_functionality(self):
        """
        Test JointParser node basic functionality.

        1. Send joint states with various joints including ignored ones
        2. Verify that ignored joints are filtered out
        3. Verify joint data is correct without requiring a specific output order
        """
        received_messages = {}

        # Generate namespace lookups for topics
        self.generate_namespace_lookup([
            'rizon_joint_states',
            'rizon_parsed_joint_states',
        ])

        # Create publisher for input joint states
        joint_state_pub = self.node.create_publisher(
            JointState, self.namespaces['rizon_joint_states'],
            qos_profile=10)

        # Create subscriber for output joint states
        subs = self.create_logging_subscribers(
            [('rizon_parsed_joint_states', JointState)],
            received_messages,
            accept_multiple_messages=True,
            qos_profile=10,
        )

        try:
            # Wait for the node to be ready
            time.sleep(1.0)

            # Create test joint state message with various joints
            input_joint_state = JointState()
            input_joint_state.header.stamp = self.node.get_clock().now().to_msg()
            input_joint_state.header.frame_id = 'base_link'
            input_joint_state.name = [
                'joint1', 'joint2', 'joint3', 'joint4',
                'joint5', 'joint6', 'joint7',
                'finger_joint', 'left_inner_finger_joint',
            ]
            input_joint_state.position = [
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
            ]
            input_joint_state.velocity = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
            ]
            input_joint_state.effort = [
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
            ]

            # Keep publishing joint state
            end_time = time.time() + TIMEOUT
            while time.time() < end_time:
                joint_state_pub.publish(input_joint_state)
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if received_messages.get('rizon_parsed_joint_states'):
                    break

            # Verify we received the output message
            self.assertIn('rizon_parsed_joint_states', received_messages)
            self.assertGreater(len(received_messages['rizon_parsed_joint_states']), 0)

            output_joint_state = received_messages['rizon_parsed_joint_states'][-1]

            # Verify header is preserved
            self.assertEqual(output_joint_state.header.frame_id, 'base_link')

            # Verify only non-ignored joints are present
            expected_joints = {
                'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
            }
            self.assertEqual(set(output_joint_state.name), expected_joints)
            self.assertEqual(len(output_joint_state.position), len(output_joint_state.name))
            self.assertEqual(len(output_joint_state.velocity), len(output_joint_state.name))
            self.assertEqual(len(output_joint_state.effort), len(output_joint_state.name))

            # Verify joint positions, velocities, and efforts are correct
            expected_joint_data = {
                name: (
                    input_joint_state.position[i],
                    input_joint_state.velocity[i],
                    input_joint_state.effort[i],
                )
                for i, name in enumerate(input_joint_state.name)
                if name in expected_joints
            }
            output_joint_data = {
                name: (
                    output_joint_state.position[i],
                    output_joint_state.velocity[i],
                    output_joint_state.effort[i],
                )
                for i, name in enumerate(output_joint_state.name)
            }
            self.assertEqual(output_joint_data, expected_joint_data)

        finally:
            for sub in subs:
                self.node.destroy_subscription(sub)
            self.node.destroy_publisher(joint_state_pub)


@pytest.mark.rostest
def generate_test_description():
    """Generate test description."""
    joint_parser_node = Node(
        name='joint_parser',
        package='isaac_ros_manipulation_flexiv_isaac_sim_utils',
        executable='isaac_sim_joint_parser_node.py',
        namespace=IsaacROSJointParserTest.generate_namespace(),
    )

    return IsaacROSJointParserTest.generate_test_description([joint_parser_node])
