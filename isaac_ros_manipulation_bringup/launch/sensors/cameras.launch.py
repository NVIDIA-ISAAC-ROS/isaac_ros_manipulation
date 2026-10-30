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
    DepthType, get_calibration_parameters, get_camera_nodes, SensorConfig
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)


def launch_setup(context, *args, **kwargs):
    sensor_config = SensorConfig(context)

    manipulator_init_nodes = []
    manipulator_init_nodes += get_camera_nodes(
        num_cameras=sensor_config.num_cameras,
        setup=sensor_config.setup,
        depth_type=sensor_config.depth_type,
        depth_engine_file_path=sensor_config.depth_engine_file_path,
        camera_type=sensor_config.camera_type,
        enable_dnn_depth_in_realsense=sensor_config.enable_dnn_depth_in_realsense,
        workflow_type=sensor_config.workflow_type)
    manipulator_init_nodes += get_calibration_parameters(sensor_config.workflow_type,
                                                         use_sim_time=sensor_config.use_sim_time,
                                                         setup=sensor_config.setup,
                                                         num_cameras=sensor_config.num_cameras,
                                                         camera_type=sensor_config.camera_type,
                                                         gripper_type=sensor_config.gripper_type,
                                                         robot_type=str(sensor_config.robot_type))
    return manipulator_init_nodes


def generate_launch_description():
    launch_args = [
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
            'workflow_type',
            choices=[
                'POSE_TO_POSE',
                'PICK_AND_PLACE',
                'OBJECT_FOLLOWING',
                'DISPLAYPORT_CABLE_INSERTION',
            ],
            description='Type of workflow to run the sim based manipulator pipeline on',
        )
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )

    return LaunchDescription(launch_args + [group_action])
