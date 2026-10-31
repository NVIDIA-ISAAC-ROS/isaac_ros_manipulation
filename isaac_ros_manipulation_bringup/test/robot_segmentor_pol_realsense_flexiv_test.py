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
"""Tests the robot segmentor for a Flexiv Rizon with Realsense setup."""

import os

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils.test_utils import (
    get_params_from_config_file_set_in_env, IsaacRosManipulationRobotSegmentorPolTest
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
ISAAC_ROS_WS = os.environ.get('ISAAC_ROS_WS')
OUTPUT_DIR = os.path.join(ISAAC_ROS_WS, 'robot_segmentor_pol_realsense_flexiv_test')


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Flexiv driver and Robot Segmentor nodes for testing."""
    IsaacRosManipulationRobotSegmentorPolTest.generate_namespace()
    isaac_ros_manipulation_workflow_bringup_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'launch')
    isaac_ros_manipulation_test_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'test', 'include')
    flexiv_driver_launch_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_flexiv_driver_utils'),
        'launch')

    params = get_params_from_config_file_set_in_env(RUN_TEST)
    override_params = {
        'enable_dnn_depth_in_realsense': 'false',
        'camera_type': 'REALSENSE',
        'num_cameras': '1',
        'enable_nvblox': 'true',
        'workflow_type': 'OBJECT_FOLLOWING',
        'depth_type': 'REALSENSE',
        'robot_type': 'FLEXIV',
        'gripper_type': 'grav',
        'setup': 'flexiv_test_bench',
        'headless': 'true',
        'load_gripper': 'true',
        'start_rviz': 'false',
        'enable_rviz_visualization': 'false',
    }
    params.update(override_params)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        node_startup_delay = 12.0
        # This starts object following without object following nodes.
        # It starts the pose to pose so that we get planning calls every few seconds.
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_test_include_dir,
                 '/cumotion.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [flexiv_driver_launch_dir, '/flexiv_rizon_real_driver.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))

        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_workflow_bringup_include_dir,
                    '/sensors/cameras.launch.py']),
            launch_arguments={key: str(value) for key, value in params.items()}.items()))
    else:
        # Makes the test pass if we do not want to run on CI
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

    return IsaacRosManipulationRobotSegmentorPolTest.generate_test_description(
        run_test=RUN_TEST,
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
        use_sim_time=False,
        output_dir=OUTPUT_DIR,
        timeout_seconds_for_service_call=30.0,
        intrinsics_topics=['/camera_1/aligned_depth_to_color/camera_info'],
        input_topics=['/camera_1/aligned_depth_to_color/image_raw'],
        output_topics=['/cumotion/camera_1/world_depth'],
        time_to_run=30,
        max_latency_ms=1000,
    )
