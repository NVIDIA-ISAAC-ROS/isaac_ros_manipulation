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
"""Tests topic QOS setting for the RT-DETR and Foundation Pose workflow for Realsense setup."""

import os

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_ros_python_utils import (
    get_params_from_config_file_set_in_env
)
from isaac_ros_manipulation_ros_python_utils.test_utils import (
    IsaacROSFoundationPoseQosTest
)
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import pytest


RUN_TEST = os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'


@pytest.mark.rostest
def generate_test_description():
    """Generate launch description with Foundation Pose nodes for testing."""
    IsaacROSFoundationPoseQosTest.generate_namespace()
    isaac_ros_manipulation_test_include_dir = os.path.join(
        get_package_share_directory('isaac_ros_manipulation_bringup'),
        'test', 'include')

    params = get_params_from_config_file_set_in_env(RUN_TEST)

    # Override the camera_type and num_cameras parameters based on this test file's requirements
    params['camera_type'] = 'REALSENSE'
    params['num_cameras'] = '1'
    params['workflow_type'] = 'OBJECT_FOLLOWING'
    params['headless'] = 'true'

    # Set up container for our nodes
    test_nodes = []
    node_startup_delay = 1.0
    if RUN_TEST:
        node_startup_delay = 10.0
        test_nodes.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [isaac_ros_manipulation_test_include_dir,
                 '/perception_foundationpose_connections_test.launch.py']),
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
    return IsaacROSFoundationPoseQosTest.generate_test_description(
        run_test=RUN_TEST,
        excluded_topics=[
            '/parameter_events',
            '/rosout',
            '/clock',
            '/camera_info_segmentation'
        ],
        topics_with_sensor_data_qos=[
            '/camera_1/aligned_depth_to_color/image_raw',
            '/camera_1/color/camera_info',
            '/camera_1/color/image_raw'
        ],
        check_sensor_data_for_topics_for_only_this_node={},
        check_sensor_data_for_topics_that_have_publisher=[],
        nodes=test_nodes,
        node_startup_delay=node_startup_delay,
    )
