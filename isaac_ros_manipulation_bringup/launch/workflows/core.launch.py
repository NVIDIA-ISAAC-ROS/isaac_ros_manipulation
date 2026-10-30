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
    CameraType, CoreConfig, DepthType, get_cumotion_node,
    get_manipulation_container, get_nvblox_node, get_workflow_nodes
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction
)


def launch_setup(context, *args, **kwargs):

    core_config = CoreConfig(context)
    manipulator_init_nodes = []
    manipulator_init_nodes.append(get_manipulation_container(core_config))
    manipulator_init_nodes.append(get_cumotion_node(
        camera_type=core_config.camera_config.camera_type,
        xrdf_file_path=core_config.cumotion_config.cumotion_xrdf_file_path,
        urdf_file_path=core_config.cumotion_config.cumotion_urdf_file_path,
        distance_threshold=core_config.cumotion_config.distance_threshold,
        num_cameras=core_config.camera_config.num_cameras,
        time_sync_slop=core_config.time_sync_slop,
        use_sim_time=core_config.use_sim_time,
        setup=core_config.setup,
        workflow_type=core_config.workflow_config.workflow_type,
        read_esdf_world=core_config.enable_nvblox,
        core_config=core_config
    ))
    if core_config.enable_nvblox:
        manipulator_init_nodes.append(get_nvblox_node(
            camera_type=core_config.camera_config.camera_type,
            use_sim_time=core_config.use_sim_time,
            setup=core_config.setup,
            num_cameras=core_config.camera_config.num_cameras,
            enable_dnn_depth_in_realsense=(
                core_config.depth_estimation_config.enable_dnn_depth_in_realsense
            ),
            nvblox_global_frame=core_config.nvblox_global_frame
        ))
    workflow_nodes = get_workflow_nodes(core_config)
    return manipulator_init_nodes + workflow_nodes


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'gripper_type',
            default_value='robotiq_2f_140',
            description='Type of gripper attached to the robot arm',
            choices=['robotiq_2f_85', 'robotiq_2f_140', 'grav'],
        ),
        DeclareLaunchArgument(
            'camera_type',
            default_value=str(CameraType.REALSENSE),
            choices=['REALSENSE', 'ISAAC_SIM'],
            description='Camera sensor to use'
        ),
        DeclareLaunchArgument(
            'depth_type',
            default_value=str(DepthType.REALSENSE),
            choices=DepthType.names(),
            description=f'Depth estimation type. Choose between {", ".join(DepthType.names())}'
        ),
        DeclareLaunchArgument(
            'num_cameras',
            default_value='1',
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
            default_value='0.15',
            description='Maximum distance from a given collision sphere (in meters) at which'
                        'to mask points in the robot segmenter'
        ),
        DeclareLaunchArgument(
            'pose_estimation_input_qos',
            default_value='SENSOR_DATA',
            description='QoS input profile for pose estimation input',
        ),
        DeclareLaunchArgument(
            'pose_estimation_input_fps',
            default_value='15',
            description='FPS for input into pose estimation pipeline'
        ),
        DeclareLaunchArgument(
            'pose_estimation_dropped_fps',
            default_value='10',
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
