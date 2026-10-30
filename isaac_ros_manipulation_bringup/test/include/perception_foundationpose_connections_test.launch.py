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

"""This file is used to test the connections between RT-DETR and FoundationPose."""

from typing import List

from isaac_ros_launch_utils import GroupAction

from isaac_ros_manipulation_ros_python_utils import (
    CoreConfig, DepthType, get_foundation_pose_nodes, get_manipulation_container,
    get_object_detection_servers, WorkflowType
)

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch_ros.actions import Node


def launch_setup(context: LaunchContext, *args, **kwargs) -> List[Node]:
    """
    Test RT-DETR and FoundationPose connections.

    This includes QOS testing for the connections and making sure topics are wired correctly.

    Args
    ----
        context (LaunchContext): Launch context

    Returns
    -------
        List[Node]: List of nodes

    """
    # Set up container for our nodes
    core_config = CoreConfig(context=context)
    all_nodes = []
    all_nodes.append(get_manipulation_container(core_config))
    if core_config.workflow_type == WorkflowType.PICK_AND_PLACE:
        all_nodes += get_object_detection_servers(
            camera_config=core_config.camera_config,
            pose_estimation_config=core_config.pose_estimation_config
        )

    # Include the foundation pose launch file
    all_nodes += get_foundation_pose_nodes(
        camera_config=core_config.camera_config,
        workflow_type=core_config.workflow_type,
        pose_estimation_config=core_config.pose_estimation_config
    )
    return all_nodes


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
            'depth_type',
            choices=DepthType.names(),
            description=f'Depth estimation type. Choose between {", ".join(DepthType.names())}'
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
    return LaunchDescription(launch_args + [group_action])
