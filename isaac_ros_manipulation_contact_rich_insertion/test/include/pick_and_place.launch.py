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
from isaac_ros_launch_utils import GroupAction

from isaac_ros_manipulation_ros_python_utils import (
    CoreConfig
)
import isaac_ros_manipulation_ros_python_utils.workflows as workflow_utils_ext

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction
)


def launch_setup(context, *args, **kwargs):

    core_config = CoreConfig(context)
    manipulator_init_nodes = []
    workflow_nodes = workflow_utils_ext.get_pick_and_place_orchestrator(
        core_config.workflow_config
    )
    return manipulator_init_nodes + workflow_nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'gripper_type',
            description='Type of gripper to use with UR robot',
            choices=['robotiq_2f_85', 'robotiq_2f_140'],
        ),
        DeclareLaunchArgument(
            'camera_type',
            choices=['REALSENSE', 'ISAAC_SIM'],
            description='Camera sensor to use'
        ),
        DeclareLaunchArgument(
            'num_cameras',
            choices=['1', '2',],
            description='Num cameras'
        ),
        DeclareLaunchArgument(
            'setup',
            description='Setup'
        ),
        DeclareLaunchArgument(
            'cumotion_urdf_file_path',
            description='URDF for cumotion planner, not the same as Moveit planner'
        ),
        DeclareLaunchArgument(
            'cumotion_xrdf_file_path',
            description='XRDF for cumotion planner, not the same as Moveit planner'
        ),
        DeclareLaunchArgument(
            'distance_threshold',
            description='Maximum distance from a given collision sphere (in meters) at which'
                        'to mask points in the robot segmenter'
        ),
        DeclareLaunchArgument(
            'pose_estimation_input_qos',
            description='QoS input profile for pose estimation input',
        ),
        DeclareLaunchArgument(
            'pose_estimation_input_fps',
            description='FPS for input into pose estimation pipeline'
        ),
        DeclareLaunchArgument(
            'pose_estimation_dropped_fps',
            description='Number of frames to drop before input into pose estimation pipeline'
        )
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )
    manipulator_init_nodes = []

    return LaunchDescription(launch_args + manipulator_init_nodes + [group_action])
