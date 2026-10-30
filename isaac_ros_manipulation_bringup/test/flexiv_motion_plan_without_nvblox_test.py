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
"""Smoke test Flexiv cuMotion planning with nvblox disabled."""

import os
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from isaac_ros_cumotion_interfaces.action import MotionPlan
from isaac_ros_manipulation_ros_python_utils.test_utils import (
    get_params_from_config_file_set_in_env
)
import launch
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import launch_testing
import pytest
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
MOTION_PLAN_ACTION = 'cumotion/motion_plan'
ENABLE_NVBLOX = False
JOINT_STATES_TOPIC = '/joint_states'
ARM_JOINT_NAME_CANDIDATES = []


def _get_flexiv_arm_joint_candidates(params):
    robot_sn = str(params.get('robot_sn', ''))
    candidates = []
    if robot_sn:
        candidates.append([f'{robot_sn}_joint{i}' for i in range(1, 8)])
    candidates.append([f'joint{i}' for i in range(1, 8)])
    return candidates


@pytest.mark.rostest
def generate_test_description():
    """Launch Flexiv driver and cuMotion without nvblox/robot segmenter."""
    global ARM_JOINT_NAME_CANDIDATES, JOINT_STATES_TOPIC

    bringup_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'launch')
    bringup_test_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'), 'test', 'include')
    flexiv_driver_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_flexiv_driver_utils'), 'launch')

    params = get_params_from_config_file_set_in_env(RUN_TEST)
    params.update({
        'enable_dnn_depth_in_realsense': 'false',
        'camera_type': 'REALSENSE',
        'num_cameras': '1',
        'enable_nvblox': 'true' if ENABLE_NVBLOX else 'false',
        'workflow_type': 'OBJECT_FOLLOWING',
        'depth_type': 'REALSENSE',
        'robot_type': 'FLEXIV',
        'gripper_type': 'grav',
        'setup': 'flexiv_test_bench',
        'headless': 'true',
        'load_gripper': 'false',
        'start_rviz': 'false',
        'enable_rviz_visualization': 'false',
    })

    JOINT_STATES_TOPIC = str(params.get('cumotion_joint_states_topic', '/joint_states'))
    ARM_JOINT_NAME_CANDIDATES = _get_flexiv_arm_joint_candidates(params)

    test_nodes = []
    if RUN_TEST:
        # Launch the Flexiv driver first so it can materialize the prefixed cuMotion URDF/XRDF.
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [flexiv_driver_launch_dir, '/flexiv_rizon_real_driver.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [bringup_test_include_dir, '/cumotion.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [bringup_launch_dir, '/sensors/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
    else:
        test_nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'world', '--child-frame-id', 'base_link',
            ]
        ))

    return launch.LaunchDescription(
        test_nodes + [TimerAction(period=10.0, actions=[launch_testing.actions.ReadyToTest()])]
    )


class FlexivMotionPlanWithoutNvbloxTest(unittest.TestCase):
    """Send a small joint-space MotionPlan goal and require success."""

    @classmethod
    def setUpClass(cls):
        if not RUN_TEST:
            return
        rclpy.init()
        cls.node = rclpy.create_node('flexiv_motion_plan_without_nvblox_test')
        cls.latest_joint_state = None
        cls.joint_state_sub = cls.node.create_subscription(
            JointState, JOINT_STATES_TOPIC, cls._joint_state_callback, 10)
        cls.motion_plan_client = ActionClient(cls.node, MotionPlan, MOTION_PLAN_ACTION)

    @classmethod
    def tearDownClass(cls):
        if not RUN_TEST:
            return
        cls.motion_plan_client.destroy()
        cls.node.destroy_subscription(cls.joint_state_sub)
        cls.node.destroy_node()
        rclpy.shutdown()

    @classmethod
    def _joint_state_callback(cls, msg):
        cls.latest_joint_state = msg

    def _wait_for_arm_joint_state(self, timeout_sec=30.0):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            msg = self.latest_joint_state
            if msg is None:
                continue
            names = set(msg.name)
            for candidate in ARM_JOINT_NAME_CANDIDATES:
                if all(name in names for name in candidate):
                    return msg, candidate
        self.fail(f'No joint state containing Flexiv arm joints on {JOINT_STATES_TOPIC}')

    def _make_goal_state(self, joint_state, arm_joint_names):
        positions_by_name = dict(zip(joint_state.name, joint_state.position))
        goal_state = JointState()
        goal_state.header.stamp = self.node.get_clock().now().to_msg()
        goal_state.name = arm_joint_names
        goal_state.position = [float(positions_by_name[name]) for name in arm_joint_names]
        goal_state.position[0] += 0.01
        return goal_state

    def test_simple_motion_plan_succeeds(self):
        """Plan a tiny joint-space move using the live Flexiv joint state."""
        if not RUN_TEST:
            self.skipTest('ENABLE_MANIPULATOR_TESTING is not set to on_robot')

        self.assertTrue(
            self.motion_plan_client.wait_for_server(timeout_sec=120.0),
            f'{MOTION_PLAN_ACTION} action server did not become available')

        joint_state, arm_joint_names = self._wait_for_arm_joint_state()
        goal = MotionPlan.Goal()
        goal.goal_state = self._make_goal_state(joint_state, arm_joint_names)
        goal.plan_cspace = True
        goal.plan_pose = False
        goal.plan_grasp = False
        goal.use_current_state = True
        goal.use_planning_scene = False
        goal.update_esdf = ENABLE_NVBLOX
        goal.clear_esdf = False
        goal.enable_aabb_clearing = False
        goal.time_dilation_factor = 0.2

        goal_future = self.motion_plan_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, goal_future, timeout_sec=15.0)
        goal_handle = goal_future.result()
        self.assertIsNotNone(goal_handle, 'MotionPlan goal handle did not arrive')
        self.assertTrue(goal_handle.accepted, 'MotionPlan goal was rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=120.0)
        result_wrapper = result_future.result()
        self.assertIsNotNone(result_wrapper, 'MotionPlan result did not arrive')
        result = result_wrapper.result
        self.assertTrue(result.success, f'MotionPlan failed: {getattr(result, "message", "")}')
