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
"""Tests for DisplayPort insertion orchestrator manual-pose mode."""

import time
import unittest

from action_msgs.msg import GoalStatus
from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import Pose
from isaac_ros_manipulation_interfaces.action import DisplayPortCableInsertion, PickAndPlace
from isaac_ros_test import IsaacROSBaseTest
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.action import ActionClient, ActionServer


import tf2_ros


RUN_TEST = True


class DisplayPortManualPoseOrchestratorTest(IsaacROSBaseTest):
    """Validate manual-pose DisplayPort insertion without perception servers."""

    def setupServers(self):
        self._pick_and_place_goal = None
        self._pick_and_place_server = ActionServer(
            self.node, PickAndPlace, '/pick_and_place', self._pick_and_place_callback)
        self._gripper_server = ActionServer(
            self.node, ParallelGripperCommand, '/robotiq_gripper_controller/gripper_cmd',
            self._gripper_callback)
        self._displayport_action_client = ActionClient(
            self.node, DisplayPortCableInsertion, '/displayport_cable_insertion')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)

    def _pick_and_place_callback(self, goal_handle):
        self._pick_and_place_goal = goal_handle.request
        result = PickAndPlace.Result()
        result.success = True
        goal_handle.succeed()
        return result

    def _gripper_callback(self, goal_handle):
        result = ParallelGripperCommand.Result()
        result.state.effort = [0.0]
        result.state.position = list(goal_handle.request.command.position)
        result.reached_goal = True
        result.stalled = False
        goal_handle.succeed()
        return result

    def test_manual_pose_workflow_publishes_tfs_and_picks(self):
        self.setupServers()
        time.sleep(3.0)

        if not self._displayport_action_client.wait_for_server(timeout_sec=10.0):
            self.fail('DisplayPortCableInsertion action server not available')

        goal_future = self._displayport_action_client.send_goal_async(
            DisplayPortCableInsertion.Goal())
        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=30.0)
        goal_handle = goal_future.result()
        self.assertTrue(goal_handle.accepted)

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=30.0)
        result = result_future.result()
        self.assertEqual(result.status, GoalStatus.STATUS_SUCCEEDED)
        self.assertTrue(result.result.success)

        self.assertIsNotNone(self._pick_and_place_goal)
        self.assertEqual(self._pick_and_place_goal.object_id, 0)
        self.assertTrue(self._pick_and_place_goal.keep_gripper_closed_after_completion)
        self.assertAlmostEqual(self._pick_and_place_goal.place_pose.position.z, 0.18)

        time.sleep(1.0)
        plug_transform = self.tf_buffer.lookup_transform(
            'base_link', 'detected_object1', rclpy.time.Time())
        self.assertAlmostEqual(plug_transform.transform.translation.x, 0.45, places=3)
        self.assertAlmostEqual(plug_transform.transform.translation.y, -0.05, places=3)
        self.assertAlmostEqual(plug_transform.transform.translation.z, 0.12, places=3)

        socket_transform = self.tf_buffer.lookup_transform(
            'base_link', 'displayport_socket_frame', rclpy.time.Time())
        self.assertAlmostEqual(socket_transform.transform.translation.x, 0.55, places=3)
        self.assertAlmostEqual(socket_transform.transform.translation.y, 0.05, places=3)
        self.assertAlmostEqual(socket_transform.transform.translation.z, 0.10, places=3)


@pytest.mark.rostest
def generate_test_description():
    if not RUN_TEST:
        return DisplayPortManualPoseOrchestratorTest.generate_test_description(
            nodes=[Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_transform_publisher',
                output='screen',
                arguments=[
                    '--x', '0', '--y', '0', '--z', '0',
                    '--yaw', '0', '--pitch', '0', '--roll', '0',
                    '--frame-id', 'world', '--child-frame-id', 'base_link',
                ]
            )],
            node_startup_delay=1.0,
        )

    return DisplayPortManualPoseOrchestratorTest.generate_test_description(
        nodes=[Node(
            package='isaac_ros_manipulation_contact_rich_insertion',
            executable='displayport_insertion_orchestrator.py',
            name='displayport_insertion_orchestrator',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'wait_for_point_topic': False,
                'use_manual_poses': True,
                'manual_plug_pose': [0.45, -0.05, 0.12, 0.0, 0.0, 0.0, 1.0],
                'manual_socket_pose': [0.55, 0.05, 0.10, 0.0, 0.0, 0.0, 1.0],
                'manual_grasp_mode': False,
                'run_rl_inference': False,
                'offset_for_hover_pose': 0.08,
                'base_frame': 'base_link',
                'socket_pose_base_frame': 'base_link',
                'socket_tf_frame': 'displayport_socket_frame',
                'gripper_action_name': '/robotiq_gripper_controller/gripper_cmd',
                'gripper_close_position': 0.65,
                'target_joint_state_for_place_pose': [0.0] * 7,
                'joint_names': [
                    'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7',
                ],
            }],
        )],
        node_startup_delay=3.0,
    )


class TestDisplayPortManualPoseHelpers(unittest.TestCase):
    """Pure helper tests that do not require ROS graph startup."""

    def test_hover_pose_z_offset(self):
        socket_pose = Pose()
        socket_pose.position.z = 0.10
        socket_pose.orientation.w = 1.0
        hover_pose = Pose()
        hover_pose.position.z = socket_pose.position.z + 0.08
        hover_pose.orientation = socket_pose.orientation
        self.assertAlmostEqual(hover_pose.position.z, 0.18)


if __name__ == '__main__':
    unittest.main()
